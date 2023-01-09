---
title: "Ernie Layout"
categories:
  - NLP
classes: wide
excerpt: Addition to Layoutlmv3.

---


## Why?
1. SOTA 
2. Pretraining objectives looks very reasonable

## What Nobel things done
1. Uses Layoutparser to find regions -- serialize this regions -- use them for Reading Order Prediction task in pretraining, unlike most models which were using Raster Scale kind of sequence

not nobel but different from v3 and v2
1. spatial-aware disentangled attention mechanism  like DeBERTa in the multimodal transformer, and designs a replaced regions prediction pre-training task, to facilitate the fine-grained interaction across textual, visual, and layout modalities.

2. separate embedding layers in the horizontal and vertical directions bbox coordinates -- Like DETR - TSR

![pretraining task](/images/ernie.png)

## Inputs
1. Text
2. BBox
3. Image

### Text embeddings
1. BERT like 
2. [CLS] and [SEP] are appended at the beginning and end of the text sequence, respectively. Finally, the text embedding of token sequence T is expressed as: `T = token embedding + 1D position embedding + token type embedding`
3. Length of textual tokens - max_length (512) here

```python
def _calc_text_embeddings(self, input_ids, bbox, position_ids,
                              token_type_ids):
        words_embeddings = self.embeddings.word_embeddings(input_ids)
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._cal_spatial_position_embeddings(
            bbox)
        token_type_embeddings = self.embeddings.token_type_embeddings(
            token_type_ids)
        embeddings = words_embeddings + position_embeddings + x1 + y1 + x2 + y2 + w + h + token_type_embeddings

        embeddings = self.embeddings.LayerNorm(embeddings)
        embeddings = self.embeddings.dropout(embeddings)
        return embeddings
```


### Visual embeddings

1. document image is resized to 224×224 and
fed into `Resnet101`,
2. `adaptive pooling layer`
is introduced to convert the output into a feature
map with a fixed width W and height H (here, we
set them to 7). 
3. flatten the feature map into
a visual sequence V , and project each visual token to the same dimension as text embedding with a linear layer 
`V = visual token embedding + 1D position embedding + token type embedding`

4. visual token length -  HW ( 7*7) = 49

```python
def _calc_img_embeddings(self, image, bbox, position_ids):
        if image is not None:
            visual_embeddings = self.visual_act_fn(
                self.visual_proj(self.visual(image.astype(paddle.float32))))
        position_embeddings = self.embeddings.position_embeddings(position_ids)
        x1, y1, x2, y2, h, w = self.embeddings._cal_spatial_position_embeddings(
            bbox)
        if image is not None:
            embeddings = visual_embeddings + position_embeddings + x1 + y1 + x2 + y2 + w + h
        else:
            embeddings = position_embeddings + x1 + y1 + x2 + y2 + w + h

        if self.has_visual_segment_embedding:
            embeddings += self.visual_segment_embedding
        embeddings = self.visual_LayerNorm(embeddings)
        embeddings = self.visual_dropout(embeddings)
        return embeddings
```

### Layout embeddings

1. all the coordinate
values are normalized in the range [0, 1000].
2. (x0, y0, x1, y1, w, h),
3. To look up the layout embeddings of textual/visual token, we construct separate
embedding layers in the horizontal and vertical directions:
L = E2x(x0, x1, w) + E2y(y0, y1, h), (3)
where E2x is the x-axis embedding layer, E2y denotes the y-axis embedding layer.
4. 


### Forwad

```python

input_shape = paddle.shape(input_ids)
visual_shape = list(input_shape)
visual_shape[1] = self.config["image_feature_pool_shape"][
    0] * self.config["image_feature_pool_shape"][1]

## same as v3

visual_bbox = self._calc_visual_bbox(
    self.config["image_feature_pool_shape"], bbox, visual_shape)

## final bbox
final_bbox = paddle.concat([bbox, visual_bbox], axis=1)


if attention_mask is None:
    attention_mask = paddle.ones(input_shape)

visual_attention_mask = paddle.ones(visual_shape)

attention_mask = attention_mask.astype(visual_attention_mask.dtype)

## final attent mask
final_attention_mask = paddle.concat(
    [attention_mask, visual_attention_mask], axis=1)

if token_type_ids is None:
    token_type_ids = paddle.zeros(input_shape, dtype=paddle.int64)

if position_ids is None:
    seq_length = input_shape[1]
    position_ids = self.embeddings.position_ids[:, :seq_length]
    position_ids = position_ids.expand(input_shape)

visual_position_ids = paddle.arange(0, visual_shape[1]).expand(
    [input_shape[0], visual_shape[1]])

## final position ids
final_position_ids = paddle.concat([position_ids, visual_position_ids],
                                    axis=1)

if bbox is None:
    bbox = paddle.zeros(input_shape + [4])

text_layout_emb = self._calc_text_embeddings(
    input_ids=input_ids,
    bbox=bbox,
    token_type_ids=token_type_ids,
    position_ids=position_ids,
)

visual_emb = self._calc_img_embeddings(
    image=image,
    bbox=visual_bbox,
    position_ids=visual_position_ids,
)

# final embedding
final_emb = paddle.concat([text_layout_emb, visual_emb], axis=1)

extended_attention_mask = final_attention_mask.unsqueeze(1).unsqueeze(2)

extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

if head_mask is not None:
    if head_mask.dim() == 1:
        head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(
            -1).unsqueeze(-1)
        head_mask = head_mask.expand(self.config["num_hidden_layers"],
                                        -1, -1, -1, -1)
    elif head_mask.dim() == 2:
        head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
else:
    head_mask = [None] * self.config["num_hidden_layers"]

encoder_outputs = self.encoder(
    final_emb,
    extended_attention_mask,
    bbox=final_bbox,
    position_ids=final_position_ids,
    head_mask=head_mask,
    output_attentions=output_attentions,
    output_hidden_states=output_hidden_states,
)
sequence_output = encoder_outputs[0]
pooled_output = self.pooler(sequence_output)
return sequence_output, pooled_output
```

## Encoder Attention

image.png
![Ernie Attn](/images/ernie_attn.png)

## Pretraining Tasks

### ROP ( Reading Order Prediction)
1. 

### RRP ( Replaced Region Prediction)
1. 10% of the image patches are randomly selected and replaced with a patch from another image
2. Then, the [CLS] vector output by the transformer
is used to predict which patches are replaced

same as v3
### MVLM (Masked Visual Language Modelling)
### TIA ( Text Image Alignment)



## Fine Tuning
```
Dataset       |Epoch    |Wt-Decay |Batch
FUNSD         |100      |-          |2
CORD          |30       |0.05       |16
SROIE         |100      |0.05       |16
Kleister-NDA  |30       |0.05       |16
RVL-CDIP      |20       |0.05       |16
DocVQA        |6        |0.05       |16
## Hyper-parameters for downstream tasks
```

![ernie ablation study](/images/ernie_ablation.png)
