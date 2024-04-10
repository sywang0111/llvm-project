//===- LinalgToEmitCPass.cpp - Linalg to EmitC Pass ---------------*- C++ -*-===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file implements a pass to convert the Linalg dialect to the EmitC
// dialect.
//
//===----------------------------------------------------------------------===//

#include "mlir/Conversion/LinalgToEmitC/LinalgToEmitCPass.h"

#include "mlir/Conversion/LinalgToEmitC/LinalgToEmitC.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/EmitC/IR/EmitC.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"

namespace mlir {
#define GEN_PASS_DEF_CONVERTLINALGTOEMITC
#include "mlir/Conversion/Passes.h.inc"
} // namespace mlir

using namespace mlir;

namespace {
struct ConvertLinalgToEmitC
    : public impl::ConvertLinalgToEmitCBase<ConvertLinalgToEmitC> {
  void runOnOperation() override;
};
} // namespace

void ConvertLinalgToEmitC::runOnOperation() {
  ConversionTarget target(getContext());

  target.addLegalDialect<emitc::EmitCDialect>();
  target.addIllegalDialect<linalg::LinalgDialect>();

  RewritePatternSet patterns(&getContext());

  TypeConverter typeConverter;
  typeConverter.addConversion([](Type type) { return type; });

  populateLinalgToEmitCPatterns(typeConverter, patterns);

  if (failed(
          applyPartialConversion(getOperation(), target, std::move(patterns))))
    signalPassFailure();
}
