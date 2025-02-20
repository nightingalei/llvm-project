//===- MlirOptMain.cpp - MLIR Optimizer Driver ----------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This is a utility that runs an optimization pass and prints the result back
// out. It is designed to support unit testing.
//
//===----------------------------------------------------------------------===//

#include "mlir/Tools/mlir-opt/MlirOptMain.h"
#include "mlir/Bytecode/BytecodeWriter.h"
#include "mlir/IR/AsmState.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/Location.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Support/DebugCounter.h"
#include "mlir/Support/FileUtilities.h"
#include "mlir/Support/Timing.h"
#include "mlir/Support/ToolUtilities.h"
#include "mlir/Tools/ParseUtilties.h"
#include "llvm/Support/CommandLine.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/InitLLVM.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/StringSaver.h"
#include "llvm/Support/ThreadPool.h"
#include "llvm/Support/ToolOutputFile.h"

using namespace mlir;
using namespace llvm;

/// Perform the actions on the input file indicated by the command line flags
/// within the specified context.
///
/// This typically parses the main source file, runs zero or more optimization
/// passes, then prints the output.
///
static LogicalResult performActions(raw_ostream &os, bool verifyDiagnostics,
                                    bool verifyPasses, SourceMgr &sourceMgr,
                                    MLIRContext *context,
                                    PassPipelineFn passManagerSetupFn,
                                    bool emitBytecode, bool implicitModule) {
  DefaultTimingManager tm;
  applyDefaultTimingManagerCLOptions(tm);
  TimingScope timing = tm.getRootScope();

  // Disable multi-threading when parsing the input file. This removes the
  // unnecessary/costly context synchronization when parsing.
  bool wasThreadingEnabled = context->isMultithreadingEnabled();
  context->disableMultithreading();

  // Prepare the parser config, and attach any useful/necessary resource
  // handlers. Unhandled external resources are treated as passthrough, i.e.
  // they are not processed and will be emitted directly to the output
  // untouched.
  PassReproducerOptions reproOptions;
  FallbackAsmResourceMap fallbackResourceMap;
  ParserConfig config(context, &fallbackResourceMap);
  reproOptions.attachResourceParser(config);

  // Parse the input file and reset the context threading state.
  TimingScope parserTiming = timing.nest("Parser");
  OwningOpRef<Operation *> op =
      parseSourceFileForTool(sourceMgr, config, implicitModule);
  context->enableMultithreading(wasThreadingEnabled);
  if (!op)
    return failure();
  parserTiming.stop();

  // Prepare the pass manager, applying command-line and reproducer options.
  PassManager pm(context, OpPassManager::Nesting::Implicit,
                 op.get()->getName().getStringRef());
  pm.enableVerifier(verifyPasses);
  applyPassManagerCLOptions(pm);
  pm.enableTiming(timing);
  if (failed(reproOptions.apply(pm)) || failed(passManagerSetupFn(pm)))
    return failure();

  // Run the pipeline.
  if (failed(pm.run(*op)))
    return failure();

  // Print the output.
  TimingScope outputTiming = timing.nest("Output");
  if (emitBytecode) {
    BytecodeWriterConfig writerConfig(fallbackResourceMap);
    writeBytecodeToFile(op.get(), os, writerConfig);
  } else {
    AsmState asmState(op.get(), OpPrintingFlags(), /*locationMap=*/nullptr,
                      &fallbackResourceMap);
    op.get()->print(os, asmState);
    os << '\n';
  }
  return success();
}

/// Parses the memory buffer.  If successfully, run a series of passes against
/// it and print the result.
static LogicalResult
processBuffer(raw_ostream &os, std::unique_ptr<MemoryBuffer> ownedBuffer,
              bool verifyDiagnostics, bool verifyPasses,
              bool allowUnregisteredDialects, bool preloadDialectsInContext,
              bool emitBytecode, bool implicitModule,
              PassPipelineFn passManagerSetupFn, DialectRegistry &registry,
              llvm::ThreadPool *threadPool) {
  // Tell sourceMgr about this buffer, which is what the parser will pick up.
  SourceMgr sourceMgr;
  sourceMgr.AddNewSourceBuffer(std::move(ownedBuffer), SMLoc());

  // Create a context just for the current buffer. Disable threading on creation
  // since we'll inject the thread-pool separately.
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  if (threadPool)
    context.setThreadPool(*threadPool);

  // Parse the input file.
  if (preloadDialectsInContext)
    context.loadAllAvailableDialects();
  context.allowUnregisteredDialects(allowUnregisteredDialects);
  if (verifyDiagnostics)
    context.printOpOnDiagnostic(false);
  context.getDebugActionManager().registerActionHandler<DebugCounter>();

  // If we are in verify diagnostics mode then we have a lot of work to do,
  // otherwise just perform the actions without worrying about it.
  if (!verifyDiagnostics) {
    SourceMgrDiagnosticHandler sourceMgrHandler(sourceMgr, &context);
    return performActions(os, verifyDiagnostics, verifyPasses, sourceMgr,
                          &context, passManagerSetupFn, emitBytecode,
                          implicitModule);
  }

  SourceMgrDiagnosticVerifierHandler sourceMgrHandler(sourceMgr, &context);

  // Do any processing requested by command line flags.  We don't care whether
  // these actions succeed or fail, we only care what diagnostics they produce
  // and whether they match our expectations.
  (void)performActions(os, verifyDiagnostics, verifyPasses, sourceMgr, &context,
                       passManagerSetupFn, emitBytecode, implicitModule);

  // Verify the diagnostic handler to make sure that each of the diagnostics
  // matched.
  return sourceMgrHandler.verify();
}

LogicalResult mlir::MlirOptMain(raw_ostream &outputStream,
                                std::unique_ptr<MemoryBuffer> buffer,
                                PassPipelineFn passManagerSetupFn,
                                DialectRegistry &registry, bool splitInputFile,
                                bool verifyDiagnostics, bool verifyPasses,
                                bool allowUnregisteredDialects,
                                bool preloadDialectsInContext,
                                bool emitBytecode, bool implicitModule) {
  // The split-input-file mode is a very specific mode that slices the file
  // up into small pieces and checks each independently.
  // We use an explicit threadpool to avoid creating and joining/destroying
  // threads for each of the split.
  ThreadPool *threadPool = nullptr;

  // Create a temporary context for the sake of checking if
  // --mlir-disable-threading was passed on the command line.
  // We use the thread-pool this context is creating, and avoid
  // creating any thread when disabled.
  MLIRContext threadPoolCtx;
  if (threadPoolCtx.isMultithreadingEnabled())
    threadPool = &threadPoolCtx.getThreadPool();

  auto chunkFn = [&](std::unique_ptr<MemoryBuffer> chunkBuffer,
                     raw_ostream &os) {
    return processBuffer(os, std::move(chunkBuffer), verifyDiagnostics,
                         verifyPasses, allowUnregisteredDialects,
                         preloadDialectsInContext, emitBytecode, implicitModule,
                         passManagerSetupFn, registry, threadPool);
  };
  return splitAndProcessBuffer(std::move(buffer), chunkFn, outputStream,
                               splitInputFile, /*insertMarkerInOutput=*/true);
}

LogicalResult mlir::MlirOptMain(raw_ostream &outputStream,
                                std::unique_ptr<MemoryBuffer> buffer,
                                const PassPipelineCLParser &passPipeline,
                                DialectRegistry &registry, bool splitInputFile,
                                bool verifyDiagnostics, bool verifyPasses,
                                bool allowUnregisteredDialects,
                                bool preloadDialectsInContext,
                                bool emitBytecode, bool implicitModule) {
  auto passManagerSetupFn = [&](PassManager &pm) {
    auto errorHandler = [&](const Twine &msg) {
      emitError(UnknownLoc::get(pm.getContext())) << msg;
      return failure();
    };
    return passPipeline.addToPipeline(pm, errorHandler);
  };
  return MlirOptMain(outputStream, std::move(buffer), passManagerSetupFn,
                     registry, splitInputFile, verifyDiagnostics, verifyPasses,
                     allowUnregisteredDialects, preloadDialectsInContext,
                     emitBytecode, implicitModule);
}

LogicalResult mlir::MlirOptMain(int argc, char **argv, llvm::StringRef toolName,
                                DialectRegistry &registry,
                                bool preloadDialectsInContext) {
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));

  static cl::opt<bool> splitInputFile(
      "split-input-file",
      cl::desc("Split the input file into pieces and process each "
               "chunk independently"),
      cl::init(false));

  static cl::opt<bool> verifyDiagnostics(
      "verify-diagnostics",
      cl::desc("Check that emitted diagnostics match "
               "expected-* lines on the corresponding line"),
      cl::init(false));

  static cl::opt<bool> verifyPasses(
      "verify-each",
      cl::desc("Run the verifier after each transformation pass"),
      cl::init(true));

  static cl::opt<bool> allowUnregisteredDialects(
      "allow-unregistered-dialect",
      cl::desc("Allow operation with no registered dialects"), cl::init(false));

  static cl::opt<bool> showDialects(
      "show-dialects", cl::desc("Print the list of registered dialects"),
      cl::init(false));

  static cl::opt<bool> emitBytecode(
      "emit-bytecode", cl::desc("Emit bytecode when generating output"),
      cl::init(false));

  static cl::opt<bool> noImplicitModule{
      "no-implicit-module",
      cl::desc(
          "Disable implicit addition of a top-level module op during parsing"),
      cl::init(false)};

  InitLLVM y(argc, argv);

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  DebugCounter::registerCLOptions();
  PassPipelineCLParser passPipeline("", "Compiler passes to run");

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);

  if (showDialects) {
    llvm::outs() << "Available Dialects:\n";
    interleave(
        registry.getDialectNames(), llvm::outs(),
        [](auto name) { llvm::outs() << name; }, "\n");
    return success();
  }

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return failure();
  }

  if (failed(MlirOptMain(output->os(), std::move(file), passPipeline, registry,
                         splitInputFile, verifyDiagnostics, verifyPasses,
                         allowUnregisteredDialects, preloadDialectsInContext,
                         emitBytecode, /*implicitModule=*/!noImplicitModule)))
    return failure();

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return success();
}
