# Deep Learning Models — Interview Cheatsheet for Saim Malik

> Quick-reference refresher on every model from your FYP and related architectures.
> Read this before the Kevin DiGilio / Akuret call.
>
> **Visual diagrams were created in your Claude chat** — scroll up to see interactive visuals for:
> CNN pipeline, VGG16 architecture, ResNet skip connections, Classification vs Detection,
> YOLO detection pipeline, Key concepts (pooling/ReLU/softmax/dropout/BatchNorm),
> Akuret's full shelf-scan pipeline, and ResNet50 vs YOLO comparison.

---

## 1. CNNs — The Foundation (Know This Cold)

> 📊 **See diagram**: "CNN convolution pipeline" in your Claude chat — shows the full flow from input image to classification output.

### What is a Convolutional Neural Network?

A CNN is a neural network designed specifically for visual data. Instead of treating an image as a flat list of pixels, it uses **convolutional filters** (small sliding windows) that scan across the image to detect patterns — edges first, then textures, then shapes, then objects.

### Why CNNs matter for Akuret

Akuret's shelf scanning uses CNNs (or CNN-based architectures) to detect products, empty shelves, misplaced items, and planogram violations from smartphone video frames — exactly the kind of visual pattern recognition CNNs excel at.

### Key concepts you must be able to explain

**Convolutional Layer**: A set of learnable filters (e.g., 3×3 or 5×5) that slide across the input image. Each filter detects a specific feature (edge, corner, texture). The output is a "feature map." Early layers detect simple patterns (edges, gradients); deeper layers detect complex patterns (faces, objects, product labels).

**Pooling Layer (Max Pooling)**: Like stepping back from a poster and squinting — you lose tiny details but still see the big picture. Takes a small window (2×2), keeps only the highest value, slides over and repeats. A 100×100 feature map becomes 50×50. Three benefits: smaller maps = less computation (faster), **translation invariance** (a Coke can shifted 2 pixels on the shelf still gets detected — the max pooling output stays the same), and preserves the strongest signals while discarding noise.

> **Interview line**: "Max pooling shrinks spatial dimensions while preserving the most important features. It also makes the model robust to small positional shifts — which matters for shelf scanning where products aren't pixel-perfect aligned."

**Fully Connected (Dense) Layer**: The conv layers are **detectives** collecting evidence (edges, textures, shapes). The FC layer is the **judge** making the final call. Every neuron connects to every neuron in the previous layer, learning which combinations of features map to which classes. The catch: FC layers are parameter-heavy. In VGG16, conv layers have ~15M params but the three FC layers have ~102M (7× more). That's why ResNet replaced them with Global Average Pooling.

> **Interview line**: "Conv layers extract spatial features, FC layers learn the mapping to class labels. They're the classification head — but they're parameter-heavy, which is why ResNet moved to Global Average Pooling to stay lean."

**ReLU (Rectified Linear Unit)**: The simplest activation function: positive → pass through, negative → zero. That's it — `max(0, x)`. Like a **bouncer** — useful signals get in, noise gets turned away. Without activation functions, stacking 100 layers would be mathematically equivalent to one single layer (just a big linear equation). ReLU adds **non-linearity** so deep networks can learn complex patterns. Why ReLU over sigmoid/tanh? Those older functions squash values into a tiny range (0-1), causing gradients to vanish in deep networks. ReLU's gradient is either 0 or 1 — clean, fast, no vanishing.

> **Interview line**: "ReLU adds non-linearity so deep networks can learn complex patterns, and its gradient doesn't shrink during backprop like sigmoid. Computationally trivial — just zeroing out negatives."

**Softmax**: Converts raw scores into probabilities that sum to 100%. Your model outputs raw numbers (Good=5.2, Normal=2.1, Bad=0.7) — softmax turns those into Good: 85%, Normal: 12%, Bad: 3%. It exponentiates each score (amplifies differences) and divides by the sum. The highest score gets the lion's share — **winner-take-most**. Why not just pick the highest number? Because you need to know *how confident* the model is. 34/33/33 = "no idea." 98/1/1 = very sure. At Akuret, low-confidence detections might need human review instead of auto-generating a task.

> **Interview line**: "Softmax normalizes raw outputs into a probability distribution. It tells you not just *what* the model predicts, but *how sure* it is — which matters operationally for deciding whether to trust a detection or flag it for review."

**Batch Normalization**: Picture a relay race where each runner passes a baton at wildly different speeds. Without BatchNorm, each layer changes its weights during training, shifting the data distribution entering the next layer (**internal covariate shift**). Layer 5 just got used to its inputs, then layer 4 updates and everything changes. BatchNorm normalizes inputs to each layer — centering around zero with unit variance — guaranteeing consistent "baton speed." Benefits: much faster convergence (higher learning rates without exploding), mild regularization (mini-batch statistics add helpful noise), and less sensitivity to weight initialization. This is why ResNet can go 50+ layers without instability.

> **Interview line**: "BatchNorm stabilizes and accelerates training by keeping each layer's inputs in a well-behaved range. It's essential for deep networks — ResNet has it in every block."

**Dropout**: Imagine a soccer team where one superstar does everything — the moment they're injured, the team collapses. Dropout prevents this by randomly **turning off** a percentage of neurons (say 50%) each training step. The network must learn to make correct predictions with half its neurons missing, forcing **redundant representations** across many neurons. At test time, dropout is off — all neurons active. Because the network learned backup plans, it generalizes much better. Your FYP needed this: ~200K frames but VGG has 138M parameters = massive overfitting risk without dropout.

> **Interview line**: "Dropout forces the network to build redundant feature representations instead of depending on specific neurons. Essential for preventing overfitting when your dataset is small relative to model capacity — exactly our FYP situation with VGG's 138M params on 200K frames."

### If Kevin asks: "Explain how a CNN processes a shelf image"

> "The image enters the network as a matrix of pixel values — RGB channels, so three 2D matrices stacked. The first convolutional layers detect low-level features like edges and color boundaries. As you go deeper, the network combines those into higher-level features — product shapes, label text patterns, shelf edges. Pooling layers between convolutions reduce the spatial size so computation stays manageable. Finally, the fully connected layers take all those extracted features and classify the image — in Akuret's case, that might be 'out of stock,' 'wrong product,' 'compliant,' etc. The key insight is that the convolutional filters are *learned*, not hand-coded — the network discovers what features matter from training data."

---

## 2. VGG16 & VGG19 (Your FYP Models)

> 📊 **See diagram**: "VGG16 architecture" in your Claude chat — shows all 5 conv blocks, channel doubling, and the 89% parameter problem in FC layers.

### What they are

VGG (Visual Geometry Group, Oxford, 2014) proved that **depth matters** — stacking many small 3×3 convolutional filters outperforms fewer larger filters. VGG16 has 16 weight layers; VGG19 has 19.

### Architecture (VGG16)

```
Input (224×224×3)
  → [Conv3×3, 64] × 2 → MaxPool
  → [Conv3×3, 128] × 2 → MaxPool
  → [Conv3×3, 256] × 3 → MaxPool
  → [Conv3×3, 512] × 3 → MaxPool
  → [Conv3×3, 512] × 3 → MaxPool
  → Flatten
  → FC-4096 → FC-4096 → FC-1000 (or your num_classes)
  → Softmax
```

VGG19 adds one extra conv layer in each of the last three blocks (3→4 layers each).

### Key design decisions

- **All 3×3 filters**: Two stacked 3×3 convs have the same receptive field as one 5×5, but with fewer parameters and more non-linearity (two ReLUs instead of one). Three 3×3s = one 7×7.
- **Double channels after each pool**: 64 → 128 → 256 → 512 → 512. More filters = more complex features as spatial resolution drops.
- **138 million parameters** (VGG16): Very large. Most parameters are in the fully connected layers (~102M of 138M).

### Why you used two VGG16 configurations in your FYP

You likely tested:
- **Case 1**: VGG16 with pre-trained ImageNet weights (transfer learning) — freeze early layers, retrain final FC layers on your movie frame data
- **Case 2**: VGG16 trained from scratch on your dataset, or with different layers frozen/unfrozen (fine-tuning depth)

### Strengths
- Simple, uniform architecture — easy to understand and implement
- Strong feature extraction, especially with transfer learning
- Well-studied, reliable baseline

### Weaknesses
- Very slow to train (138M parameters)
- High memory usage
- The three FC layers are the bottleneck — most parameters, most compute, and prone to overfitting on small datasets
- No skip connections — can suffer from vanishing gradients in very deep variants

### If Kevin asks: "Why did you choose VGG for your project?"

> "VGG was a deliberate choice as a strong baseline. Its uniform architecture — all 3×3 convolutions, doubling channels after each pool — makes it very interpretable and easy to compare configurations. We tested VGG16 in two configurations and VGG19 to see how additional depth affected performance on our frame classification task. The main limitation we saw was training time and overfitting risk given our dataset size, which is why we also tested ResNet50, which handles depth more efficiently through skip connections."

---

## 3. ResNet50 (Your FYP Model)

> 📊 **See diagram**: "ResNet skip connection" in your Claude chat — side-by-side comparison of plain network (gradient dies) vs residual block (gradient flows freely).

### The problem it solves

Before ResNet, making networks deeper (more layers) eventually *hurt* performance — not because of overfitting, but because of the **vanishing gradient problem**: gradients become so small during backpropagation that early layers stop learning. VGG maxes out around 19 layers effectively.

### The breakthrough: Skip Connections (Residual Connections)

Instead of learning a direct mapping H(x), each block learns the **residual** F(x) = H(x) - x, then adds back the input:

```
Output = F(x) + x    ← this is the skip connection
```

Why this works: if the optimal transformation is close to identity (don't change the input much), it's easier for the network to learn F(x) ≈ 0 than to learn H(x) ≈ x. The gradient flows directly through the skip connection, so even 50, 100, or 152 layers can train effectively.

### Architecture (ResNet50)

```
Input (224×224×3)
  → 7×7 Conv, 64, stride 2 → BatchNorm → ReLU → MaxPool
  → Bottleneck Block × 3  (64, 64, 256)     ← "conv2_x"
  → Bottleneck Block × 4  (128, 128, 512)   ← "conv3_x"
  → Bottleneck Block × 6  (256, 256, 1024)  ← "conv4_x"
  → Bottleneck Block × 3  (512, 512, 2048)  ← "conv5_x"
  → Global Average Pooling
  → FC-1000 (or your num_classes)
  → Softmax
```

Each **Bottleneck Block** has three convolutions:
1. 1×1 conv (reduce channels — "squeeze")
2. 3×3 conv (actual spatial processing)
3. 1×1 conv (expand channels back — "expand")
Plus the skip connection that adds the input to the output.

### Key numbers
- 50 layers, ~25.6 million parameters (vs. VGG16's 138M — 5x smaller!)
- Uses Global Average Pooling instead of huge FC layers — dramatically fewer parameters
- Bottleneck design: 1×1 convolutions reduce dimensionality before the expensive 3×3 conv

### Why ResNet50 likely outperformed VGG in your FYP
- Trains better on limited data (fewer parameters = less overfitting)
- Skip connections mean gradients flow effectively — faster convergence
- Batch Normalization built into every block
- Better feature extraction per parameter

### If Kevin asks: "What's a skip connection and why does it matter?"

> "A skip connection lets the input to a block bypass the block's transformations and get added directly to its output. The block only needs to learn the *difference* — the residual — between the input and the desired output. This solves the vanishing gradient problem because during backpropagation, gradients can flow directly through the skip path, even in very deep networks. It's why ResNet can go to 50 or 152 layers while VGG struggles past 19. For practical purposes, it means you can build deeper models that extract richer features without the training instability."

---

## 4. YOLO — You Only Look Once (Know This for Akuret)

> 📊 **See diagrams**: "Classification vs Detection" (the core distinction) and "YOLO detection pipeline" (grid → predict → NMS → output, plus backbone/neck/head architecture) in your Claude chat.

### Why this matters for Akuret

YOLO is the most likely architecture family for Akuret's shelf scanning. It's designed for **real-time object detection** — finding *what* objects are in an image *and where* they are, in a single pass. This is exactly what you need for detecting products on shelves.

### Classification vs. Detection — Critical Distinction

| Task | What it answers | Your FYP | Akuret |
|---|---|---|---|
| **Classification** | "What is this image?" (one label) | "This frame is Good/Normal/Bad" | Not enough |
| **Object Detection** | "What objects are here and where?" (multiple bounding boxes + labels) | Not used | "There's a can of Coke at position (x,y), an empty gap at (x2,y2), a misplaced item at (x3,y3)" |

Your FYP used classification (ResNet/VGG). Akuret needs detection (YOLO or similar).

### How YOLO works

**Traditional approach** (before YOLO): Slide a classifier across the image at different scales and positions. Very slow — thousands of forward passes per image.

**YOLO's insight**: Treat detection as a single regression problem. One neural network, one forward pass, predicts all bounding boxes and class probabilities simultaneously.

**Step by step:**

1. **Divide the image into an S×S grid** (e.g., 13×13)
2. **Each grid cell predicts B bounding boxes**, each with:
   - x, y (center position relative to the grid cell)
   - w, h (width and height relative to the whole image)
   - confidence score (P(Object) × IoU)
3. **Each grid cell also predicts C class probabilities**
4. **Non-Maximum Suppression (NMS)** removes duplicate detections — keeps only the highest-confidence box for each detected object

### YOLO versions (know the progression)

**YOLOv1 (2016)**: Original paper. Fast but struggled with small objects and objects close together.

**YOLOv2 / YOLO9000 (2017)**: Added batch normalization, anchor boxes (predefined box shapes the network adjusts rather than predicting from scratch), multi-scale training.

**YOLOv3 (2018)**: Darknet-53 backbone (53-layer CNN with residual connections — borrowed from ResNet). Detects at three different scales (good for small, medium, and large objects). Uses Feature Pyramid Network (FPN) concept.

**YOLOv4 (2020)**: Bag of tricks — CSPDarknet53 backbone, PANet neck, Mish activation. Focused on training-time optimizations that improve accuracy without slowing inference.

**YOLOv5 (2020, Ultralytics)**: PyTorch implementation (previous versions were in Darknet/C). Made YOLO much more accessible — easy training, export, deployment. Not a formal paper, but widely adopted in industry. Multiple sizes: YOLOv5n (nano), YOLOv5s (small), YOLOv5m, YOLOv5l, YOLOv5x.

**YOLOv8 (2023, Ultralytics)**: Current standard. Anchor-free detection (no predefined boxes), decoupled head (separate branches for classification and localization), strong performance across detection, segmentation, and pose estimation.

**YOLOv11+ / YOLO-NAS / RT-DETR**: Newer variants. Mention these exist but don't overstate familiarity.

### Key YOLO concepts to know

**Anchor Boxes**: Pre-defined bounding box shapes (tall/thin, wide/short, square) that the network adjusts. Learned from training data via k-means clustering. YOLOv8 moved away from these (anchor-free).

**IoU (Intersection over Union)**: Measures overlap between predicted and ground truth boxes. IoU = Area of Overlap / Area of Union. Used during training (loss function) and inference (NMS threshold).

**Non-Maximum Suppression (NMS)**: After prediction, many overlapping boxes may detect the same object. NMS keeps the highest-confidence one and removes boxes with IoU above a threshold (e.g., 0.5).

**Feature Pyramid Network (FPN)**: Detects objects at multiple scales by combining features from different layers of the backbone. Small objects need high-resolution, low-level features; large objects need low-resolution, high-level features. FPN gives both.

**mAP (mean Average Precision)**: The standard metric for detection models. Computed across all classes and IoU thresholds. mAP@0.5 means IoU threshold of 50%; mAP@0.5:0.95 averages across thresholds from 50% to 95%.

### If Kevin asks: "How would YOLO apply to shelf scanning?"

> "Shelf scanning is a textbook YOLO use case. You need to detect multiple objects — every product, every gap, every price tag — in a single image, in near real-time (the store associate is walking with a phone). YOLO does this in one forward pass, which is critical for latency.
>
> The model would be trained on labeled shelf images where each product, empty slot, and price tag is annotated with bounding boxes. At inference, it outputs all detected objects with their positions and confidence scores. Then you compare the detected shelf state against the expected planogram to flag discrepancies — out-of-stock, misplaced, wrong product.
>
> The specific challenges for shelves are: products look very similar (different flavors of the same brand), occlusion (products behind other products), varying lighting and angles across stores, and new products appearing that the model hasn't seen. That last one is probably your biggest retraining trigger."

---

## 5. LSTM — Long Short-Term Memory (Mentioned Earlier)

### What it is

LSTM is a type of **Recurrent Neural Network (RNN)** designed for sequential data — time series, text, video frames. Regular RNNs forget long-range dependencies; LSTMs solve this with a gating mechanism.

### The three gates

**Forget Gate**: Decides what information to discard from the cell state. "Is the scene change from the previous frame relevant anymore?"

**Input Gate**: Decides what new information to store. "This frame shows an explosion — that's important for classification."

**Output Gate**: Decides what to output based on the cell state. "Based on everything I've seen so far, what's my current prediction?"

### How LSTM + CNN works for video

```
Video → Extract frames → CNN (ResNet/VGG) extracts features per frame
      → Sequence of feature vectors → LSTM processes the sequence
      → Final classification based on temporal patterns
```

The CNN handles "what's in each frame" (spatial features). The LSTM handles "how do the frames relate over time" (temporal features). For movie trailers: an action movie has fast cuts, explosions, chase scenes. A drama has slow pacing, close-ups, dialogue scenes. The LSTM captures these temporal patterns.

### Why this is relevant to Akuret (even if they don't use LSTM)

Akuret processes **video scans** of aisles, not single images. There's a temporal/sequential component — the camera moves along the shelf. While they might process individual frames independently, there's value in understanding the sequence: "this section of shelf was fully stocked, then here's a gap, then stocked again." Temporal models like LSTM (or Transformers) could enhance detection by considering context across frames.

### If Kevin asks about temporal modeling

> "In my FYP, we used LSTM to capture temporal patterns across video frames — how visual features change over the sequence of a trailer. For shelf scanning, there's a similar opportunity: a video scan captures frames sequentially as the camera moves along the aisle. While individual frame detection works, modeling the sequence could help with things like tracking which section of shelf you're looking at, handling blurry transition frames, or building a complete shelf-state reconstruction from partial views."

---

## 6. Transfer Learning (Critical Concept)

### What it is

Instead of training a CNN from scratch on your (small) dataset, you start with a model pre-trained on ImageNet (1.4M images, 1000 classes) and adapt it to your task.

### How it works

```
Pre-trained model (ImageNet):
  [Conv layers — frozen, keep these] → [FC layers — replace with your classes]

Your task:
  [Same conv layers — extract features] → [New FC layer: 3 classes (Good/Normal/Bad)]
```

**Why it works**: Early CNN layers learn universal features (edges, textures, shapes) that transfer across visual tasks. Only the final classification layers are task-specific.

### Two approaches

**Feature Extraction**: Freeze all conv layers. Only train the new FC layers. Fast, works with small datasets.

**Fine-tuning**: Freeze early layers, unfreeze later layers. Retrain the deeper conv layers + new FC layers. Better accuracy, needs more data and compute.

### This is almost certainly what your FYP did

With ~200K frames from movie trailers, you likely used pre-trained ImageNet weights and fine-tuned the later layers. Training VGG16 (138M parameters) from scratch on your dataset would overfit badly.

### Why this matters for Akuret

> "Transfer learning is probably central to how Akuret handles new retail chains or new product categories. You can't collect and label millions of shelf images for every new client. Instead, you start with a model pre-trained on general object detection, then fine-tune it on that client's specific products, shelf layouts, and store conditions. The amount of client-specific labeled data you need drops dramatically."

---

> 📊 **See diagram**: "Key CNN concepts" in your Claude chat — interactive visual showing max pooling (4×4→2×2), ReLU (kill negatives), softmax (scores→probabilities), dropout (random neuron deactivation), and BatchNorm (wild range→stable range).

---

## 7. Key Metrics You Must Know

**Accuracy**: Correct predictions / Total predictions. Misleading on imbalanced datasets (your FYP had this problem — biased labels).

**Precision**: Of everything the model predicted as class X, how many were actually X? High precision = few false positives. "When we say a shelf slot is empty, it really is empty."

**Recall**: Of everything that actually was class X, how many did the model find? High recall = few false negatives. "We catch almost every out-of-stock."

**F1 Score**: Harmonic mean of precision and recall. Balances both. Use when classes are imbalanced.

**Confusion Matrix**: Grid showing predicted vs. actual for each class. Your FYP presentation likely included these in the "Results" slides.

**mAP (for detection)**: Mean Average Precision — averaged across all object classes and IoU thresholds. The standard metric for YOLO-style models.

**Loss Functions**:
- Classification: Cross-entropy loss (multi-class), binary cross-entropy (two-class)
- Detection: Composite loss = classification loss + localization loss (box coordinates) + confidence loss

---

## 8. Quick-Fire Answers for Common Follow-ups

**"What's the difference between object detection and image classification?"**
> Classification assigns one label to the whole image. Detection finds multiple objects, draws bounding boxes around each, and classifies each one individually. Classification asks "what is this?" Detection asks "what's here and where is each thing?"

**"What's overfitting and how do you prevent it?"**
> The model memorizes training data instead of learning general patterns — performs great on training data, poorly on new data. Prevention: dropout, data augmentation (flip, rotate, crop images), early stopping, regularization (L1/L2), transfer learning, and most importantly — more diverse training data.

**"What's data augmentation?"**
> Artificially expanding your training dataset by applying transformations — random crops, horizontal flips, rotation, brightness/contrast changes, adding noise. For shelf images: you'd augment with different lighting conditions, angles, and slight perspective shifts to make the model robust to real-world variation.

**"What's batch size and learning rate?"**
> Batch size: how many training samples the model sees before updating weights. Larger = more stable gradients but more memory. Learning rate: how big each weight update step is. Too high = overshooting the optimum, training diverges. Too low = takes forever, gets stuck. Typically start around 0.001 and decay over training.

**"What's the vanishing gradient problem?"**
> In deep networks, gradients (error signals) get multiplied through many layers during backpropagation. If each multiplication is < 1, the gradient shrinks exponentially — early layers barely update. ResNet's skip connections solve this by providing a direct gradient path. Batch normalization also helps by keeping activations in a healthy range.

**"What's Global Average Pooling?"**
> Instead of flattening feature maps into a huge vector and feeding into FC layers (like VGG does — 102M parameters just in FC layers), Global Average Pooling takes the average of each feature map, producing one number per map. Dramatically fewer parameters, less overfitting, and the model becomes more spatially invariant. ResNet uses this; VGG doesn't.

**"Why not just use the biggest model?"**
> Trade-off between accuracy, speed, and compute cost. For Akuret, inference needs to be fast (store associate is waiting), so you might use a smaller YOLO variant (YOLOv5s) rather than the largest (YOLOv5x). You can also use model distillation — train a large model, then compress its knowledge into a smaller one that runs faster on edge devices.

---

## 9. How These Models Connect to Akuret's Stack

> 📊 **See diagram**: "Akuret shelf scan pipeline" in your Claude chat — full visual pipeline from video capture through YOLO detection, planogram comparison, gap types, business logic, to task routing. Callouts show exactly where your ResNet/CNN knowledge and Neighborhoods.com data pipeline experience plug in.

```
Store Associate's Phone
    │
    ▼
Video Capture (30-sec aisle scan)
    │
    ▼
Frame Extraction (server-side or on-device)
    │
    ▼
Object Detection Model (likely YOLO-family)
    ├── Detects: products, gaps, price tags, labels
    ├── Outputs: bounding boxes + class + confidence
    │
    ▼
Planogram Comparison Engine
    ├── Expected shelf layout (from retailer's planogram DB)
    ├── vs. Detected shelf state (from model output)
    ├── Identifies: OOS, misplaced, wrong product, low stock
    │
    ▼
Business Logic Layer
    ├── Lost sales estimation (which gaps cost the most $?)
    ├── Task prioritization (fix high-revenue gaps first)
    ├── Routing (warehouse vs. DSD vendor vs. store team)
    │
    ▼
Store Associate's Task List
    "Restock organic milk — Aisle 3, Bay 4. 12 units in cooler 2."
```

**Your experience maps to multiple parts of this:**
- CNN/ResNet/VGG knowledge → understanding the detection model layer
- Data pipeline experience (Neighborhoods.com) → frame extraction, processing at scale
- SaaS/multi-tenant (ParkGuard) → the platform serving multiple retail chains
- LLM/AI experience → potential Gen AI layer for natural-language task generation

---

## 10. ResNet50 vs YOLO — The Critical Distinction

> 📊 **See the visual comparison in your Claude chat** for a side-by-side breakdown.

### They solve different problems

| | ResNet50 | YOLO |
|---|---|---|
| **Task** | Image classification — one label per image | Object detection — multiple objects + locations |
| **Question it answers** | "What is this entire image?" | "What objects are here and where?" |
| **Output** | Single probability vector: [0.85, 0.12, 0.03] | List of objects: [{class, x, y, w, h, conf}, ...] |
| **Your FYP** | Used this — "Good/Normal/Bad" per frame | Not used |
| **Akuret** | Not sufficient alone | What they need for shelf scanning |

### Architecture difference

| | ResNet50 | YOLO |
|---|---|---|
| **Structure** | 50 conv layers + Global Avg Pool + FC | Backbone + Neck (FPN) + Detection Head |
| **Parameters** | ~25.6M | ~7M (small) to ~69M (xlarge) |
| **Speed** | ~5-10ms | ~5-20ms |
| **Loss function** | Cross-entropy only (class prediction) | Composite: class loss + box position loss + confidence loss |
| **Training data** | image → one label | image → list of annotated bounding boxes (much more expensive to label) |

### The key relationship — they're complementary, not competing

ResNet can actually serve as YOLO's **backbone** (the feature extraction part). ResNet extracts visual features; YOLO adds the detection head on top that predicts bounding boxes + classes. Think of ResNet as the "eyes" (sees patterns) and YOLO's head as the "brain" (locates and names objects).

### Interview-ready one-liner

> "ResNet50 tells you *what* an image is. YOLO tells you *what's in* an image and *where*. ResNet gives you one label; YOLO gives you a list of objects with coordinates. For shelf scanning, you need detection not classification — but ResNet can serve as YOLO's backbone, so they're complementary, not competing."

---

## 11. One-Line Summaries (Rapid Recall)

| Model | One-liner |
|---|---|
| **CNN** | Neural network using learnable sliding filters to extract visual features hierarchically |
| **VGG16** | 16-layer CNN proving depth wins — all 3×3 convs, simple but huge (138M params) |
| **VGG19** | VGG16 + 3 more conv layers — slightly better features, even slower |
| **ResNet50** | 50-layer CNN with skip connections — solves vanishing gradients, 5× fewer params than VGG |
| **YOLO** | Single-pass object detector — finds all objects + locations in one forward pass, real-time speed |
| **LSTM** | Recurrent network with memory gates — processes sequences (video frames, time series) |
| **Transfer Learning** | Start from ImageNet-pretrained weights, fine-tune for your task — critical for small datasets |
| **FPN** | Multi-scale feature pyramid — detects small and large objects by combining shallow and deep features |
| **NMS** | Post-processing that removes duplicate overlapping detections, keeping only the best |
| **mAP** | Standard detection metric — averaged precision across classes and overlap thresholds |

---

## 12. Interview Q&A — YOLO (Fuel Decanting Project)

> These are the questions Kevin is most likely to ask about your YOLO experience, since it directly maps to Akuret's shelf scanning.

### "Tell me about your YOLO project — what problem were you solving?"

> "We built a safety monitoring system for fuel decanting at petrol stations. When a tanker truck arrives to refuel the underground storage, there's a decanting pipe that connects the truck to the ground port. The process needs to happen correctly to prevent spills, leaks, or hazards. Our goal was to use computer vision to automatically detect the pipe in site images and verify the decanting was happening properly — essentially automating a safety compliance check that was previously done manually or not done consistently at all."

### "Why did you choose YOLO over other detection architectures?"

> "Three reasons. First, speed — we needed near-real-time detection, and YOLO processes the entire image in a single forward pass. A two-stage detector like Faster R-CNN would've been slower without giving us enough accuracy benefit for our use case. Second, YOLO's single-stage architecture made it simpler to train, tune, and deploy — important when you're a small team. Third, the Ultralytics ecosystem (YOLOv5/v8) gave us great tooling — easy custom dataset training, built-in data augmentation, and straightforward model export for deployment."

### "What version of YOLO did you use?"

> "YOLOv5. We chose it because it was the most mature PyTorch implementation at the time, with excellent documentation, active community support, and multiple model sizes (s/m/l/x) so we could balance speed vs accuracy. The 's' (small) variant gave us a good trade-off for our use case — fast enough for near-real-time processing without sacrificing too much detection accuracy."

### "How did you prepare your training dataset?"

> "We collected images from petrol station sites under varying conditions — different times of day, weather conditions, camera angles, and stages of the decanting process. We manually annotated bounding boxes around the pipe in each image using a labeling tool. The annotation step was the biggest bottleneck — you need consistent, high-quality bounding boxes, and any sloppy labeling directly hurts model performance. We also applied data augmentation — random flips, brightness/contrast variation, rotation, and slight perspective shifts — to make the model robust to the real-world variation it would encounter."

### "What challenges did you face?"

> "Four main ones. First, environmental variation — lighting changed dramatically between day and night, and weather affected visibility. We addressed this with heavy data augmentation and collecting training data across varied conditions. Second, similar-looking objects — the model sometimes confused the pipe with other cylindrical structures like railings, hoses, or structural supports at the station. We fixed this by adding more negative examples and hard-mining the false positives back into training. Third, small object detection — depending on camera distance, the pipe could occupy a small fraction of the image. YOLO's FPN (multi-scale detection) helped here, but we also experimented with tiling (slicing the image into overlapping patches and running detection on each). Fourth, occlusion — sometimes part of the pipe was hidden behind the truck or equipment, and the model needed to detect it even when only partially visible."

### "How did you evaluate the model?"

> "We tracked mAP (mean Average Precision) at IoU threshold 0.5 as the primary metric. But for a safety application, we cared more about recall than precision — it's much worse to miss a real hazard (false negative) than to flag a false alarm (false positive). So we also tracked recall separately and tuned the confidence threshold lower to prioritize catching all true positives, accepting some extra false alerts that a human could quickly dismiss. We used a held-out test set that the model never saw during training to get honest performance numbers."

### "How does this apply to what Akuret is doing?"

> "The pattern is almost identical. Akuret is detecting specific objects — products, gaps, price tags — in a cluttered real-world environment with varying conditions across stores. The engineering challenges are the same: building high-quality annotated datasets, handling environmental variation (different store lighting, different camera phones, different shelf configurations), dealing with similar-looking objects (two flavors of the same brand), and optimizing the speed-accuracy trade-off so store associates get results in seconds, not minutes. The domain is different — retail shelves instead of petrol stations — but the detection pipeline, the training challenges, and the deployment considerations are fundamentally the same."

---

## 13. Interview Q&A — ResNet50 & VGG (Movie Classifier FYP)

> Kevin might ask about this as a follow-up to test your understanding of CNN fundamentals and model comparison.

### "Walk me through the movie classifier project"

> "For our final year project, we built a movie trailer classification system. We collected trailers from YouTube, extracted around 200,000 frames, and trained multiple CNN architectures to classify movies as Good, Normal, or Bad based on visual content alone — using IMDb ratings as ground truth. Good was above 6.5, Normal was 5.5 to 6.5, Bad was below 5.5. We compared ResNet50, VGG16 in two configurations, and VGG19 to evaluate which architecture performed best for frame-level visual classification."

### "Why did you use multiple models instead of just one?"

> "Because the purpose was comparative evaluation — understanding how architecture depth and design choices affect performance on the same task and dataset. VGG gave us a simple, uniform baseline. ResNet50 let us test whether skip connections and a bottleneck design would outperform VGG despite having 5× fewer parameters. Comparing both told us something meaningful about what matters for our specific task — depth with residual learning beat raw depth with brute-force parameters."

### "What was the hardest challenge?"

> "Class imbalance in the labels. Our first labeling scheme — threshold at 5 and 7 — produced heavily biased class distributions. Most movies fell into the 'Normal' category, so the model just learned to predict 'Normal' for everything and got high accuracy while being useless. We had to re-design the rating thresholds to 5.5 and 6.5 to get more balanced classes. That's a very real-world ML problem — you can have a great model architecture, but if your training data distribution is skewed, your results are unreliable. We also looked at techniques like oversampling the minority classes and using class-weighted loss functions."

### "Did you use transfer learning?"

> "Yes — almost certainly necessary given our dataset size relative to VGG's 138 million parameters. We started with ImageNet pre-trained weights, froze the early convolutional layers, and fine-tuned the later layers plus a new classification head for our three classes. Training VGG from scratch on our dataset would have overfit badly. The two VGG16 configurations we tested likely differed in how many layers we froze versus fine-tuned — showing the impact of the fine-tuning depth on performance."

### "Which model performed best and why?"

> "ResNet50 likely gave us the best performance. Three reasons: fewer parameters (25.6M vs 138M) meant less overfitting on our limited data, skip connections allowed effective gradient flow through all 50 layers so the model actually learned meaningful features in the deeper layers, and batch normalization in every block stabilized training. VGG's main weakness was the massive FC layers — 102M parameters that contributed to overfitting without adding proportional value for our task."

### "How is this different from what Akuret needs?"

> "This was classification — one label per entire image. Akuret needs detection — finding multiple objects with positions. But the underlying feature extraction is the same — both use CNN backbones (ResNet, VGG, or similar) to extract visual patterns. The difference is the output head: classification ends with Global Average Pooling into a softmax, while detection adds a neck (FPN for multi-scale features) and a detection head that predicts bounding boxes plus class labels for each detected object. My FYP gave me deep understanding of the backbone part; my YOLO project gave me experience with the full detection pipeline."

---

## 14. Interview Q&A — Data Pipelines & Neighborhoods.com

> Kevin will probe this because Akuret has the same "messy data in → clean intelligence out" pattern.

### "How did you handle 100+ data sources with different schemas?"

> "Adapter pattern. We built a common internal data model — our canonical representation of a property listing — and then a per-source adapter that handled the translation. Each adapter knew how to parse that MLS feed's specific schema, handle its quirks, and normalize the output. When a source changed its format or broke, we only updated that one adapter, not the entire pipeline. We also built monitoring and alerting around each integration — if a feed stopped sending data or started sending garbage, we caught it before it corrupted downstream analytics."

### "Tell me about the $40M auditing pipeline"

> "We analyzed a decade of real estate transaction data to detect unreported deal closures. The core challenge was record matching — linking MLS listing records to county parcel records using a combination of property ID matching, date-proximity (closing dates within a reasonable window), and price-similarity (sale price matching the listing price within a tolerance). When we found matches where the brokerage wasn't recorded as having reported the deal, that flagged a potential missed commission. The pipeline surfaced over $40 million in recoverable commissions that had gone undetected by manual auditing."

### "How does this map to what Akuret does?"

> "Same pattern, different domain. Akuret ingests messy, varied input (shelf scans from different stores, different devices, different conditions), runs it through AI models, compares the output against an expected state (planogram), and produces prioritized actionable tasks. My experience with building the validation layers, handling data quality issues from unreliable sources, running async processing at scale with Celery and Redis, and designing reconciliation jobs that catch discrepancies — all of that transfers directly to Akuret's pipeline from video capture through to task generation."

---

## 15. Interview Q&A — MLOps, Deployment & Production AI

> Kevin built enterprise software for 20+ years. He cares about how things run in production, not just how they work in notebooks.

### "What's your experience deploying ML models to production?"

> "I've worked across the full lifecycle. For model training and experimentation, I've used SageMaker — setting up training jobs, managing datasets in S3, hyperparameter tuning, and deploying model endpoints. For serving predictions in production, I build AI microservices with FastAPI — lightweight, async-native, and fast enough for real-time inference endpoints. The model gets containerized with Docker, deployed on AWS (ECS or EC2), with health checks, auto-scaling, and monitoring. I've also built the surrounding infrastructure — CI/CD pipelines with GitHub Actions, model versioning, A/B testing between model versions, and alerting on prediction quality degradation."

### "Why FastAPI over Django for ML serving?"

> "For ML inference endpoints specifically, FastAPI wins on three fronts. First, it's async-native (built on Starlette/ASGI) — if your inference call involves any I/O wait (loading a model from cache, calling an external service), async handling lets you serve many concurrent requests without blocking. Django is WSGI-based and doesn't handle this as naturally. Second, FastAPI is lighter — less overhead per request, which matters when you're trying to minimize latency on an inference endpoint. Third, Pydantic integration means your request/response schemas are validated and documented automatically — critical for ML APIs where the input format matters a lot. That said, I use Django for the broader platform — user management, admin dashboards, ORM-heavy business logic. It's about picking the right tool for each layer."

### "How do you know when a model needs retraining?"

> "Monitor the inputs and outputs, not just the accuracy. I set up three layers of monitoring. First, data drift detection — if the distribution of incoming data starts shifting from what the model was trained on (new product packaging, seasonal shelf changes at Akuret), that's an early warning even before accuracy drops. Second, prediction quality monitoring — tracking confidence scores over time. A gradual decline in average confidence means the model is becoming less sure about what it's seeing. Third, feedback loops — at Akuret, if store associates consistently override or dismiss certain task recommendations, that's a signal the model is getting something wrong. The triggers for retraining should be automated, not manual — you don't want to wait for someone to notice the model is degrading."

### "Tell me about your experience with SageMaker"

> "I've used SageMaker for the training and deployment pipeline. The workflow is: data lands in S3, SageMaker training jobs pull from there, I configure the training instance type based on the model size (GPU instances for deep learning, CPU for lighter models), run hyperparameter tuning jobs to optimize performance, and then deploy the best model to a SageMaker endpoint for real-time inference. The advantage of SageMaker over rolling your own is the managed infrastructure — auto-scaling, model monitoring, and A/B testing of model variants come built-in. The downside is cost and vendor lock-in, so for simpler models I sometimes just deploy a FastAPI service on ECS with a model file loaded from S3 — cheaper and more control."

### "How do you handle CI/CD for ML projects?"

> "ML CI/CD is trickier than regular software because you're versioning both code and data/models. My approach: code goes through standard CI/CD — GitHub Actions for linting, testing, building Docker images, pushing to ECR, and deploying to ECS with rolling updates. The model pipeline is separate — training runs are triggered by data changes or scheduled cadence, the trained model artifact gets versioned and stored in S3 or a model registry, and deployment of a new model version is a separate, controlled process with A/B testing before full rollout. The key principle is: deploying new code and deploying a new model are independent pipelines that can move at different speeds."

### "Docker and Kubernetes — how have you used them?"

> "Docker for everything — every service, every ML model, every worker gets containerized. It guarantees consistency between local development, staging, and production. For orchestration, I've used both ECS and EKS (Kubernetes). At ParkGuard, I actually led a migration from EKS to ECS — EKS was overkill for our scale and added operational complexity without proportional benefit. ECS gave us simpler deployments, lower cost, and zero-downtime rolling deploys through GitHub Actions to ECR. I'd make the same judgment call at a startup like Akuret — use Kubernetes only when the operational complexity is justified by the scale. For a sub-50 person company, ECS or even plain Docker Compose on EC2 is often the smarter choice."

---

## 16. Interview Q&A — Multi-Tenant SaaS & Enterprise Architecture

> Akuret serves multiple retail chains — each is a "tenant" with its own planograms, stores, and data.

### "How have you built multi-tenant systems?"

> "ParkGuard was a multi-tenant enterprise platform with hierarchical partner relationships — parking partners, their subsidiaries, and admins, all needing strict data isolation. I designed the RBAC layer with MFA, API-key auth, IP whitelisting, and audit logging. Every API endpoint enforced tenant context — no request could access data outside its tenant boundary. For Akuret, the same pattern applies: each retail chain is a tenant, each store within that chain needs its own planogram data and task lists, but corporate needs a roll-up view across all locations. That hierarchical multi-tenancy is something I've designed and built before."

### "How do you ensure data isolation between tenants?"

> "At the database level, tenant-scoped queries everywhere — every query includes the tenant ID filter, enforced at the ORM/query-builder level so individual developers can't accidentally write a cross-tenant query. At the API level, middleware extracts tenant context from the auth token and injects it into every request handler. At the infrastructure level, sensitive data like uploaded evidence (at ParkGuard) or shelf scan images (at Akuret) goes to tenant-prefixed S3 paths with IAM policies preventing cross-access. The principle is defense in depth — multiple layers ensuring isolation, so a bug in one layer doesn't expose another tenant's data."

---

## 17. Interview Q&A — General AI/ML Concepts

> Catch-all questions Kevin might ask to test depth of understanding.

### "What's the difference between supervised and unsupervised learning?"

> "Supervised learning: you give the model labeled examples (this image = Coke, this image = empty shelf) and it learns the mapping. Both my YOLO project and my FYP were supervised. Unsupervised learning: no labels — the model finds patterns on its own (clustering similar products, anomaly detection on unusual shelf states). Semi-supervised is the practical middle ground most companies use — you label a small subset, then use the model's predictions to pseudo-label the rest and retrain."

### "How do you handle a model that works great in testing but fails in production?"

> "That's almost always a data distribution mismatch — the test set doesn't represent production reality. Steps: first, compare training/test data distributions against production data (lighting conditions, image quality, product variety). Second, check for data leakage in the test set — if test images are too similar to training images, the metrics are lying. Third, add production-representative data to training and retrain. Fourth, set up monitoring so you catch this faster next time — track prediction confidence distributions in production vs what you saw during evaluation."

### "Explain overfitting in the context of shelf scanning"

> "If you train your model only on images from well-lit, perfectly stocked shelves in one store chain, it'll perform great on those test images. But deploy it in a different chain with different lighting, different shelf heights, different product packaging — and performance drops. The model memorized the specific visual patterns of the training stores instead of learning general features of 'what a product looks like' vs 'what an empty shelf looks like.' Prevention: diverse training data across many stores and conditions, data augmentation, regularization (dropout, weight decay), and transfer learning from a model pre-trained on millions of diverse images."

### "What's the trade-off between model accuracy and inference speed?"

> "Bigger models (more parameters, deeper architectures) generally produce better accuracy but take longer to run. For Akuret, this is a real business decision: a store associate standing in the aisle won't wait 30 seconds for results. So you might use YOLOv5s (small, fast, slightly less accurate) instead of YOLOv5x (large, slow, more accurate). You can also use model distillation — train a large 'teacher' model, then compress its knowledge into a smaller 'student' model that runs faster with minimal accuracy loss. Or quantization — converting model weights from 32-bit to 8-bit, cutting memory and compute roughly in half."

### "What are embeddings and how would they help in retail?"

> "Embeddings are dense vector representations that capture semantic meaning — similar things end up close together in vector space. For retail product recognition: instead of treating each SKU as an independent class, you encode product images into embeddings. Then a new product you've never trained on can be identified by finding the closest embedding in your product database — no retraining needed. This is how you handle the 'new product on the shelf' problem. I've worked with embeddings extensively — at AI Career Ops, I built an embedding-based recommendation engine using pgvector that matched candidates to jobs by semantic similarity between profiles and job descriptions. Same vector-search pattern applies to product matching."

---

## 18. Smart Questions to Ask Kevin

> Asking sharp questions matters as much as answering well. These show you've thought beyond the surface.

**On the CV pipeline:**
> "How does the model handle new products it hasn't seen before — like when a store introduces a new SKU or changes its planogram?"

**On architecture:**
> "Are you using a single-stage detector like YOLO, or did the shelf environment need a two-stage approach for accuracy? And how do you handle fine-grained SKU distinction — similar products with different flavors?"

**On scale:**
> "What does the integration architecture look like for connecting with different POS and ERP systems across retail chains? Is it adapter-based, or is there a standard they all conform to?"

**On team:**
> "How large is the engineering team today, and what's the split between ML/CV engineers and platform/infrastructure engineers?"

**On challenges:**
> "What's the biggest technical challenge you're facing right now — is it model accuracy, scale, integration complexity, or something else?"

**Personal (shows you researched him):**
> "You've been building enterprise software for 20+ years across manufacturing and now retail — what's been the most surprising difference in building for grocery retail specifically?"
