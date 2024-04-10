add_mlir_conversion_library(MLIRTensorToEmitC
  TensorToEmitC.cpp
  TensorToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/TensorToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRTensorDialect
  MLIREmitCDialect
  MLIRPass
  MLIRTransformUtils
  )
