add_mlir_dialect_library(MLIRArithmeticTransforms
  BufferizableOpInterfaceImpl.cpp
  Bufferize.cpp
  EmulateWideInt.cpp
  ExpandOps.cpp
  UnsignedWhenEquivalent.cpp

  ADDITIONAL_HEADER_DIRS
  {$MLIR_MAIN_INCLUDE_DIR}/mlir/Dialect/Arithmetic/Transforms

  DEPENDS
  MLIRArithmeticTransformsIncGen

  LINK_LIBS PUBLIC
  MLIRAnalysis
  MLIRArithmeticDialect
  MLIRBufferizationDialect
  MLIRBufferizationTransforms
  MLIRFuncDialect
  MLIRFuncTransforms
  MLIRInferIntRangeInterface
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRTransforms
  MLIRTransformUtils
  MLIRVectorDialect
  )
