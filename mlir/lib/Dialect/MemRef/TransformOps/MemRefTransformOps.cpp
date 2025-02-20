//===- MemRefTransformOps.cpp - Implementation of Memref transform ops ----===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.h"
#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arithmetic/IR/Arithmetic.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/MemRef/Transforms/Passes.h"
#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/Transform/IR/TransformDialect.h"
#include "mlir/Dialect/Transform/IR/TransformInterfaces.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// MemRefMultiBufferOp
//===----------------------------------------------------------------------===//

DiagnosedSilenceableFailure
transform::MemRefMultiBufferOp::applyToOne(memref::AllocOp target,
                                           SmallVector<Operation *> &results,
                                           transform::TransformState &state) {
  auto newBuffer = memref::multiBuffer(target, getFactor());
  if (failed(newBuffer)) {
    Diagnostic diag(target->getLoc(), DiagnosticSeverity::Note);
    diag << "op failed to multibuffer";
    return DiagnosedSilenceableFailure::silenceableFailure(std::move(diag));
  }

  results.push_back(newBuffer.value());
  return DiagnosedSilenceableFailure(success());
}

//===----------------------------------------------------------------------===//
// Transform op registration
//===----------------------------------------------------------------------===//

namespace {
class MemRefTransformDialectExtension
    : public transform::TransformDialectExtension<
          MemRefTransformDialectExtension> {
public:
  using Base::Base;

  void init() {
    declareDependentDialect<pdl::PDLDialect>();
    declareGeneratedDialect<AffineDialect>();
    declareGeneratedDialect<arith::ArithmeticDialect>();

    registerTransformOps<
#define GET_OP_LIST
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.cpp.inc"
        >();
  }
};
} // namespace

#define GET_OP_CLASSES
#include "mlir/Dialect/MemRef/TransformOps/MemRefTransformOps.cpp.inc"

void mlir::memref::registerTransformDialectExtension(
    DialectRegistry &registry) {
  registry.addExtensions<MemRefTransformDialectExtension>();
}
