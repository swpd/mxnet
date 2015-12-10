/*!
 * Copyright (c) 2015 by Contributors
 * \file multi-embedding.cu
 * \brief
 * \author Zeying Xie
*/

#include "./multi-embedding-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<gpu>(MultiEmbeddingParam param) {
  return new MultiEmbeddingOp<gpu>(param);
}
}  // namespace op
}  // namespace mxnet

