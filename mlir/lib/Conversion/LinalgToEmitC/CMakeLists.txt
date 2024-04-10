add_mlir_conversion_library(MLIRLinalgToEmitC
  LinalgToEmitC.cpp
  LinalgToEmitCPass.cpp

  ADDITIONAL_HEADER_DIRS
  ${MLIR_MAIN_INCLUDE_DIR}/mlir/Conversion/LinalgToEmitC

  DEPENDS
  MLIRConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRLinalgDialect
  MLIREmitCDialect
  MLIRPass
  MLIRTransformUtils
  )
