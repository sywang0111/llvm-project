#ifndef MLIR_CONVERSION_LINALGTOEMITC_LINALGTOEMITC_H
#define MLIR_CONVERSION_LINALGTOEMITC_LINALGTOEMITC_H

namespace mlir {
class RewritePatternSet;
class TypeConverter;

void populateLinalgToEmitCPatterns(TypeConverter &typeConverter,
                                  RewritePatternSet &patterns);
} // namespace mlir

#endif // MLIR_CONVERSION_LINALGTOEMITC_LINALGTOEMITC_H
