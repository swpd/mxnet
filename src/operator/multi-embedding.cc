/*!
 * Copyright (c) 2015 by Contributors
 * \file multi-embedding.cc
 * \brief
 * \author Zeying Xie
*/

#include "./multi-embedding-inl.h"
namespace mxnet {
namespace op {
template<>
Operator* CreateOp<cpu>(MultiEmbeddingParam param) {
  return new MultiEmbeddingOp<cpu>(param);
}

Operator* MultiEmbeddingProp::CreateOperator(Context ctx) const {
  DO_BIND_DISPATCH(CreateOp, param_);
}

DMLC_REGISTER_PARAMETER(MultiEmbeddingParam);

MXNET_REGISTER_OP_PROPERTY(MultiEmbedding, MultiEmbeddingProp)
.describe("Get embedding for multi-hot input")
.add_argument("data", "Symbol", "Input data to the MultiEmbeddingOp.")
.add_argument("weight", "Symbol", "MultiEmbedding weight matrix.")
.add_arguments(MultiEmbeddingParam::__FIELDS__());
}  // namespace op
}  // namespace mxnet
