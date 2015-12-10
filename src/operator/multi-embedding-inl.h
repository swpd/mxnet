/*!
 * Copyright (c) 2015 by Contributors
 * \file multi-embedding-inl.h
 * \brief
 * \author Zeying Xie
*/
#ifndef MXNET_OPERATOR_MULTI_EMBEDDING_INL_H_
#define MXNET_OPERATOR_MULTI_EMBEDDING_INL_H_

#include <dmlc/logging.h>
#include <dmlc/parameter.h>
#include <mxnet/operator.h>
#include <map>
#include <vector>
#include <string>
#include <utility>
#include "./operator_common.h"

namespace mxnet {
namespace op {

namespace multiembedding {
enum MultiEmbeddingOpInputs {kData, kWeight};
enum MultiEmbeddingOpOutputs {kOut};
}  // namespace multiembedding

struct MultiEmbeddingParam: public dmlc::Parameter<MultiEmbeddingParam> {
  int input_dim;
  int output_dim;
  DMLC_DECLARE_PARAMETER(MultiEmbeddingParam) {
    DMLC_DECLARE_FIELD(input_dim).set_lower_bound(1)
    .describe("input dim of multi-hot encoding");
    DMLC_DECLARE_FIELD(output_dim).set_lower_bound(1)
    .describe("output dim of multi-embedding");
  }
};


template<typename xpu>
class MultiEmbeddingOp : public Operator {
 public:
  explicit MultiEmbeddingOp(MultiEmbeddingParam p) {
    this->param_ = p;
  }

  virtual void Forward(const OpContext &ctx,
                       const std::vector<TBlob> &in_data,
                       const std::vector<OpReqType> &req,
                       const std::vector<TBlob> &out_data,
                       const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(req[multiembedding::kOut], kWriteTo);
    CHECK_EQ(in_data.size(), 2);
    CHECK_EQ(out_data.size(), 1);
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> data = in_data[multiembedding::kData].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> wmat = in_data[multiembedding::kWeight].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> out = out_data[multiembedding::kOut].get<xpu, 2, real_t>(s);
    out = take(data, wmat);
  }

  virtual void Backward(const OpContext &ctx,
                        const std::vector<TBlob> &out_grad,
                        const std::vector<TBlob> &in_data,
                        const std::vector<TBlob> &out_data,
                        const std::vector<OpReqType> &req,
                        const std::vector<TBlob> &in_grad,
                        const std::vector<TBlob> &aux_args) {
    using namespace mshadow;
    using namespace mshadow::expr;
    CHECK_EQ(out_grad.size(), 1);
    CHECK_GE(in_data.size(), 1);
    CHECK_EQ(in_grad.size(), 2);
    CHECK_EQ(req[multiembedding::kData], kNullOp)
      << "MultiEmbedding layer doesn't support calculate data gradient";
    Stream<xpu> *s = ctx.get_stream<xpu>();
    Tensor<xpu, 1> data = in_data[multiembedding::kData].get<xpu, 1, real_t>(s);
    Tensor<xpu, 2> grad_out = out_grad[multiembedding::kOut].get<xpu, 2, real_t>(s);
    Tensor<xpu, 2> grad_in = in_grad[multiembedding::kWeight].get<xpu, 2, real_t>(s);
    grad_in = take_grad(data, grad_out, param_.input_dim);
  }

 private:
  MultiEmbeddingParam param_;
};  // class MultiEmbeddingOp

template<typename xpu>
Operator* CreateOp(MultiEmbeddingParam param);

#if DMLC_USE_CXX11
class MultiEmbeddingProp : public OperatorProperty {
 public:
  std::vector<std::string> ListArguments() const override {
    return {"data", "weight"};
  }

  void Init(const std::vector<std::pair<std::string, std::string> >& kwargs) override {
    param_.Init(kwargs);
  }

  std::map<std::string, std::string> GetParams() const override {
    return param_.__DICT__();
  }

  bool InferShape(std::vector<TShape> *in_shape,
                  std::vector<TShape> *out_shape,
                  std::vector<TShape> *aux_shape) const override {
    using namespace mshadow;
    const TShape &dshape = (*in_shape)[multiembedding::kData];
    if (dshape.ndim() ==  0) return false;
    SHAPE_ASSIGN_CHECK(*in_shape, multiembedding::kWeight, Shape2(param_.input_dim,
                                                          param_.output_dim));
    out_shape->clear();
    out_shape->push_back(Shape2(dshape[0], param_.output_dim));
    return true;
  }

  OperatorProperty* Copy() const override {
    auto sym = new MultiEmbeddingProp();
    sym->param_ = this->param_;
    return sym;
  }

  std::string TypeString() const override {
    return "MultiEmbedding";
  }

  std::vector<int> DeclareBackwardDependency(
    const std::vector<int> &out_grad,
    const std::vector<int> &in_data,
    const std::vector<int> &out_data) const override {
    return {out_grad[multiembedding::kOut], in_data[multiembedding::kData]};
  }

  Operator* CreateOperator(Context ctx) const override;

 private:
  MultiEmbeddingParam param_;
};  // class MultiEmbeddingProp
#endif  // DMLC_USE_CXX11

}  // namespace op
}  // namespace mxnet
#endif  // MXNET_OPERATOR_MULTI_EMBEDDING_INL_H_
