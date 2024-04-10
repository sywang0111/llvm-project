//===- TensorToEmitCPass.cpp - Tensor to EmitC Pass ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Tensor dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/TensorToEmitC/TensorToEmitCPass.h"

#include "mlir/Conversion/TensorToEmitC/TensorToEmitC.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTTENSORTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertTensorToEmitC
    : public impl::ConvertTensorToEmitCBase<ConvertTensorToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertTensorToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<tensor::TensorDialect>();

  RewritePatternSet patterns(&getContext());

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

  populateTensorToEmitCPatterns(typeConverter, patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
