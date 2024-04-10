//===- TensorToEmitC.cpp - Tensor to EmitC Patterns ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements patterns to convert the  dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToEmitC/TensorToEmitC.h"

#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

enum class FunctionName {
  dim,
  extract_slice,
  insert_slice,
  unknown
};
// function_name to int mapping
FunctionName getFunctionName(StringRef func_name) {
  if (func_name == "tensor.dim") {
    return FunctionName::dim;
  } else if (func_name == "tensor.extract_slice") {
    return FunctionName::extract_slice;
  } else if (func_name == "tensor.insert_slice") {
    return FunctionName::insert_slice;
  } else {
    return FunctionName::unknown;
  }
}

template <typename TensorOp, typename EmitCOp>
class TensorOpConversion final : public OpConversionPattern<TensorOp> {
public:
  using OpConversionPattern<TensorOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(TensorOp tensorOp, typename TensorOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef func_name = tensorOp.getOperationName();
    switch(getFunctionName(func_name)) {
      case FunctionName::dim:
        func_name = "get_tensor_dim";
        break;
      case FunctionName::extract_slice:
        func_name = "copyin";
        break;
      case FunctionName::insert_slice:
        func_name = "copyout";
        break;
      case FunctionName::unknown:
        return rewriter.notifyMatchFailure(tensorOp, "unknown function name");
    }
    
    rewriter.template replaceOpWithNewOp<EmitCOp>(
      tensorOp,
      tensorOp.getType(),
      func_name,
      adaptor.getOperands());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateTensorToEmitCPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    TensorOpConversion<tensor::DimOp, emitc::CallOpaqueOp>,
    TensorOpConversion<tensor::ExtractSliceOp, emitc::CallOpaqueOp>,
    TensorOpConversion<tensor::InsertSliceOp, emitc::CallOpaqueOp>
  >(typeConverter, ctx);
  // clang-format on
}
