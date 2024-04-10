//===- LinalgToEmitC.cpp - Linalg to EmitC Patterns ---------------*- C++ -*-===//
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

#include "mlir/Conversion/LinalgToEmitC/LinalgToEmitC.h"

#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Transforms/DialectConversion.h"

using namespace mlir;

//===----------------------------------------------------------------------===//
// Conversion Patterns
//===----------------------------------------------------------------------===//

namespace {

enum class FunctionName {
  matmul,
  unknown
};
// function_name to int mapping
FunctionName getFunctionName(StringRef func_name) {
  if (func_name == "linalg.matmul") {
    return FunctionName::matmul;
  } else {
    return FunctionName::unknown;
  }
}

template <typename LinalgOp, typename EmitCOp>
class LinalgOpConversion final : public OpConversionPattern<LinalgOp> {
public:
  using OpConversionPattern<LinalgOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(LinalgOp linalgOp, typename LinalgOp::Adaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    StringRef func_name = linalgOp.getOperationName();
    switch(getFunctionName(func_name)) {
      case FunctionName::matmul:
        func_name = "matmul";
        break;
      case FunctionName::unknown:
        return rewriter.notifyMatchFailure(linalgOp, "unknown function name");
    }
    
    rewriter.template replaceOpWithNewOp<EmitCOp>(
      linalgOp,
      linalgOp.getOperands()[0].getType(),
      func_name,
      adaptor.getOperands());

    return success();
  }
};

} // namespace

//===----------------------------------------------------------------------===//
// Pattern population
//===----------------------------------------------------------------------===//

void mlir::populateLinalgToEmitCPatterns(TypeConverter &typeConverter,
                                        RewritePatternSet &patterns) {
  MLIRContext *ctx = patterns.getContext();

  // clang-format off
  patterns.add<
    LinalgOpConversion<linalg::MatmulOp, emitc::CallOpaqueOp>
  >(typeConverter, ctx);
  // clang-format on
}
