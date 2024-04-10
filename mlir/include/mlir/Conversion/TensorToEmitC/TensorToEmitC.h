#ifndef MLIR_CONVERSION_TENSORTOEMITC_TENSORTOEMITC_H
#define MLIR_CONVERSION_TENSORTOEMITC_TENSORTOEMITC_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

void populateTensorToEmitCPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_TENSORTOEMITC_TENSORTOEMITC_H
