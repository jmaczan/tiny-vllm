# tiny-vllm

You're going to build a high performance LLM inference engine with C++ and CUDA - tiny-vllm, a younger and smaller sibling of [vLLM](https://github.com/vllm-project/vllm)

We will learn a lot along the way, make mistakes and derive the ideas and maths from scratch

This repository consists of two things: 1. a full source code of the inference server and 2. a course where I lead you through the process of implementing the engine. Feel invited to use it as a learning tool on your learning path or if you are a lecturer, feel welcome to use it as a teaching resource at your university

The inference engine consists of:

- [x] load a real LLM model from Safetensors (Llama 3.2 1B Instruct)
- [x] full LLM forward pass (prefill + decode)
- [x] all computation with CUDA kernels
- [x] KV cache
- [x] static batching
- [x] continuous batching
- [x] [online softmax, FlashAttention-like](https://courses.cs.washington.edu/courses/cse599m/23sp/notes/flashattn.pdf)
- [x] [PagedAttention](https://arxiv.org/pdf/2309.06180)

Make yourself a hot beverage and let's begin

- [tiny-vllm](#tiny-vllm)
  - [Intro: LLM, vLLM, models, inference servers](#intro-llm-vllm-models-inference-servers)
  - [Technical prerequisities](#technical-prerequisities)
  - [Safetensors and your model](#safetensors-and-your-model)
  - [How floating-point numbers work and why we use bfloat16](#how-floating-point-numbers-work-and-why-we-use-bfloat16)
  - [GPU and CPU memory](#gpu-and-cpu-memory)
  - [Single token inference](#single-token-inference)
  - [Tokenization](#tokenization)
  - [Embeddings](#embeddings)
  - [CUDA kernel engineering - embeddings](#cuda-kernel-engineering---embeddings)
  - [RMSNorm and parallel reduction in CUDA](#rmsnorm-and-parallel-reduction-in-cuda)
  - [RoPE](#rope)
  - [Residual connections](#residual-connections)
  - [cublasGemmEx](#cublasgemmex)
  - [The column-major to row-major transposition trick](#the-column-major-to-row-major-transposition-trick)
  - [Prefill vs decode](#prefill-vs-decode)
  - [Why KV cache exists](#why-kv-cache-exists)
  - [Attention](#attention)
  - [GQA](#gqa)
  - [SiLU](#silu)
  - [Softmax](#softmax)
  - [Causal mask](#causal-mask)
  - [Argmax](#argmax)
  - [Feed forward network](#feed-forward-network)
  - [Buffer reuse](#buffer-reuse)
  - [Static batching](#static-batching)
  - [Continuous batching](#continuous-batching)
  - [Online softmax](#online-softmax)
  - [Paged Attention](#paged-attention)
  - [Paged KV cache](#paged-kv-cache)
  - [Paged Attention CUDA kernel](#paged-attention-cuda-kernel)

## Intro: LLM, vLLM, models, inference servers

It's easy to get lost with so much going on recent years. Let's unpack it

**LLM is a model**. Physically, **LLM is a file which contains a lot of [float numbers](https://en.wikipedia.org/wiki/Floating-point_arithmetic)**. Conceptually, these numbers represent weights of operations. Weights are learned/discovered/found during training phase. Some of the operations use these weights. Every operation is a function, which takes some data as input, do something with it and produces data as output. Operations and their order are defined by LLM's architecture. Every model has its own architecture, which is designed by engineers and researchers.

The process of going from 0 to LLM writing a text is like this:

0. **Design the model** - engineers and researchers use high level language like Python with tensor library like [PyTorch](https://github.com/pytorch/pytorch) or [tinygrad](https://github.com/tinygrad/tinygrad) to design model's architecture. They train small versions of the model, make experiments with different operations, data and hyperparameters (parameters for operations). It's the phase of figuring out the specification
1. **Implement the model** - Once they decide on final model architecture and prepare the data for training, they write the code that defines the final model. It can be also in PyTorch or similar
2. **Train the model** - The chosen model architecture is initialized with dummy weights. They write a script which again uses PyTorch or similar to run learning algorithm like backpropagation on a lot of hardware, like [GPUs](https://en.wikipedia.org/wiki/Graphics_processing_unit) and [TPUs](https://en.wikipedia.org/wiki/Tensor_Processing_Unit). This phase burns a lot of energy, money and computational power. The product of training phase is a file with model weights, in some format, like [Safetensors format](https://huggingface.co/docs/safetensors/index). So, the training phase is finding such a set of weights which produces good text using the given architecture
3. **Serve the model (we are here)** - The file with weights can't be ran on a computer. It's not an executable. It's a lot of numbers. The architecture can't be ran either - it's just a plan, a blueprint, a description of computation. To actually run the model, we need a program that turns the architecture and its operations into executable code and uses file with model weights to load the weights into the architecture. Once you write a program that implements the operations and once the program loads the weights (weights are loaded in the runtime of the program, at the startup), you can finally send prompts to the model and get a meaningful response. Generating an output from a model is called inference. That's why what we build here is called an inference server or inference engine

Knowing the reason behind a need for an inference server, let's think why we build it in C++ and CUDA. It's because we want to maximize efficient use of the hardware and get high performance. It means that we want to get responses fast and we want to be able to handle multiple prompts at the same time. CUDA is the whole ecosystem, but also a language that you use to write code that runs on GPUs. We need to write code on GPUs, because many operations inside LLM are multiplying and adding multiple numbers. If you need to do small amount of math, CPU enough. If a lot, GPU better. LLMs are mostly about multiplying the matrices, which boils down to computing dot products of two vectors, for many numbers and for many vectors. The math of LLMs is simple, we will need basics of linear algebra and you can learn while coding and fill the gaps on the go. I find this way of JIT learning the most effective and perhaps you will like it too

My take on a relationship between AI and computation which you maybe find useful is that **the intelligence comes from a lot of parameters of the model and a lot of computation of input values using these parameters**. There is no a single element, that you can point to and say: "this is what makes the model intelligent or useful". Every part of the model you can replace with a different one and get different tradeoffs in return, like trade accuracy for complexity.  I hope I won't forget to get back to this topic later, when we touch the math of attention. Because - the default attention mechanism is very computationally complex (O(n^2*d)). And this complexity can be challenged and in fact people do it and figure out alternative attention mechanisms, like [linear attention](https://haileyschoelkopf.github.io/blog/2024/linear-attn/). If more people will find this course useful, I will think about another one, about ML compilers (a practical one in Python or C++ + some SSA theory) or about alternative attention mechanisms (math + CUDA kernels). If you are interested, please let me know! If you will find this course valuable, please let other people know about it

> Out of scope: The training phase of an LLM is something we don't do in this course. We take a trained LLM and write a program which will run this LLM fast on NVIDIA GPU for multiple requests in parallel. If you want to train your own LLM, I strongly recommend sensei Karpathy repositories like [nanoGPT](https://github.com/karpathy/nanoGPT) and [llm.c](https://github.com/karpathy/llm.c) and his [YouTube channel](https://www.youtube.com/@AndrejKarpathy). Similarly, we don't design the model, but the tensor libraries are also fascinating topic and worth understanding from scratch. George Hotz's [tinygrad](https://github.com/tinygrad/tinygrad) is a project which implements a tensor library with a very little amount of code, so if you want to get inspired and learn the internals, it's a good place to do it (also [their Discord is nice](https://discord.com/invite/ZjZadyC7PK))! There is also a bit older and smaller version by Andrej Karpathy - [micrograd](https://github.com/karpathy/micrograd). And since I brought the Discord, I want to recommend you [Mark Saroufim's](https://www.marksaroufim.com/) [GPU MODE](https://discord.com/invite/gpumode). Many great people hanging out there! And if you feel lost with what is going on here, and you are new on your AI/ML journey, start with [Jeremy Howard and Rachel Thomas](https://www.fast.ai/about) [fastai book](https://course.fast.ai/Resources/book.html). I conveniently omit the data science and engineering part here, because I don't know much about it. Probably [Kaggle](https://www.kaggle.com/) can be a good place to start with it and learn on-hands. Last but not least, we're going to code in C++ and CUDA and use [cuBLAS](https://developer.nvidia.com/cublas) where applicable. You can learn on the go. NVIDIA [official resources](https://docs.nvidia.com/cuda/cuda-programming-guide/) are good and helpful 

## Technical prerequisities

You can build and run it on any platform, with minor changes, assuming you have a NVIDIA GPU. You might need to adjust some paths, like CUDA or GCC in [c_cpp_propertiesjson](.vscode/c_cpp_properties.json) or NVCC in [CMakeLists.txt](CMakeLists.txt)

I suggest you to fork this repo and make the necessary adjustments so it works on your machine and then create a pull request to [jmaczan/tiny-vllm](https://github.com/jmaczan/tiny-vllm) and upstream your changes for benefit of another readers

The exact setup on which I develop and test it:
- Linux (6.19.8 x64_64)
- [CUDA Toolkit](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/) (13.1)
- C++ 17
- [GCC](https://gcc.gnu.org/) (15.2.1)
- The only external dependency you will pull in is JSON parser [nlohmann/json](https://github.com/nlohmann/json) 3.12.0, which is a single header file [include/json.hpp](include/json.hpp)
- AMD CPU (Ryzen 7 9800X3D)
- NVIDIA GPU (RTX 5090)
- I used [Llama 3.2 1B Instruct](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) from Hugging Face (commit hash 898999bd25b40516fce5a5b8f0948f4c81c650bc), you need just `model.safetensors` file from this repository

Install the dependencies and run the program with `./test.sh` - it will build it and immediately execute it

It also runs on AMD GPUs through ROCm/HIP. Pass `-DUSE_HIP=ON` to CMake and it builds with hipcc against hipBLAS instead of nvcc and cuBLAS; the CUDA sources are reused as-is through a thin `src/cuda_to_hip.h` compatibility header. Pick your GPU's architecture with `-DCMAKE_HIP_ARCHITECTURES` (for example `gfx90a` for MI200, `gfx1100` for RDNA3, `gfx1201` for RDNA4) - it is not hardcoded, so set it to match your card:

```bash
cmake -B build -DUSE_HIP=ON -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DCMAKE_PREFIX_PATH=/opt/rocm -G Ninja
cmake --build build
```

The `-DCMAKE_PREFIX_PATH=/opt/rocm` lets CMake find the hip and hipBLAS packages; drop it if `/opt/rocm/bin` is already on your `PATH`, or change it if ROCm lives elsewhere. I tested the AMD path on gfx90a, gfx1100, and gfx1201. The default build (no `-DUSE_HIP`) is unchanged and still targets NVIDIA through CUDA.

If you fail to build or run it and your AI of choice won't be able to help, please open an Issue on GitHub - I will try to help. Make sure to provide all useful context

## Safetensors and your model

First thing you need to do is to download a LLM which we will use to run inference on. I choose Llama 3.2 1B Instruct, because it's easy, small, tuned for dialogs and good enough for us. From perspective of us, the engineers who build an inference server, the model is just a single file containing weights.

The model is in [Safetensors format](https://huggingface.co/docs/safetensors/index). There exist other formats, like [Pickle](https://docs.python.org/3/library/pickle.html) and [Parquet](https://parquet.apache.org/docs/file-format/). Safetensors is just very popular and widely used, and the model we picked is hosted in Safetensors

Let's stop for a moment and understand the Safetensors format before we move on.

A safetensor file consists of 3 sections, always in this order: header size, header and tensors data. Header size is always 8 bytes. These 8 bytes are an unsigned 64-bit integer, which says how many bytes the actual header takes.

```cpp
std::ifstream safetensors_file("model.safetensors", std::ios_base::binary);
uint64_t header_size;
safetensors_file.read(reinterpret_cast<char *>(&header_size), 8);
```

The header is a JSON that contains about all the tensors inside the file. JSON is just a group of pairs <key, value>, where key is a unique string with a tensor name and value is another JSON object, containing info about this tensor. Every key in this JSON is a name of the tensor, except a single key which is called `__metadata__`, probably for some additonal info when necessary (we won't use it, specs say it's a "special key for storing free form text-to-text map). Every value is a JSON containing three keys - `dtype`, `shape` and `offsets`. `dtype` says what data type the tensor is stored in. `shape` says the [dimensions of a tensor](https://en.wikipedia.org/wiki/Tensor#As_multidimensional_arrays) and `offsets` say where the tensor is stored, within the tensors data section. Every `shape` is a list of ints of unknown length and every `offsets` value is a vector of exactly two ints. First element says where the tensor begins and last element says where the tensor ends.

You face the first design decision now. Do you want to make your inference server architecture independent, so it can run any arbitrary model, as long as you implement the operations it needs, or do you want to start simple and focus on our model of choice?

Whatever you decide, it's always easier to develop on a single model and then generalize, than try to make it flexible from the beginning when you're not sure how the code will look like at the end. You can always get back to it and update when you choose to.

If you want to make your server model-independent, you need to allocate the memory, set the model data `shape` and type (`dtype`) dynamically based on Safetensors header and implement more operations, making sure to cover all operations used by all models that you want to support. Probably you would still need to provide some blueprints of the models architecture, because the Safetensors file doesn't tell you which operation to run with this data, in what order etc. I'm not sure what's the optimal approach to do that, but if you figure it out, either on your own or by reading through vLLM/[TensorRT](https://github.com/NVIDIA/TensorRT)/others code, feel invited to share your findings

I will assume that we just code our server for Llama 3.2 1B Instruct architecture. Here's a dump of [Hugging Face Transformers](https://huggingface.co/docs/transformers/index) `LlamaForCausalLM` object with `meta-llama/Llama-3.2-1B-Instruct` model loaded. We can inspect which operations we need to implement and what data shape and data types we need to use:

```py
LlamaForCausalLM(
  (model): LlamaModel(
    (embed_tokens): Embedding(128256, 2048)
    (layers): ModuleList(
      (0-15): 16 x LlamaDecoderLayer(
        (self_attn): LlamaAttention(
          (q_proj): Linear(in_features=2048, out_features=2048, bias=False)
          (k_proj): Linear(in_features=2048, out_features=512, bias=False)
          (v_proj): Linear(in_features=2048, out_features=512, bias=False)
          (o_proj): Linear(in_features=2048, out_features=2048, bias=False)
        )
        (mlp): LlamaMLP(
          (gate_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (up_proj): Linear(in_features=2048, out_features=8192, bias=False)
          (down_proj): Linear(in_features=8192, out_features=2048, bias=False)
          (act_fn): SiLUActivation()
        )
        (input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
        (post_attention_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
      )
    )
    (norm): LlamaRMSNorm((2048,), eps=1e-05)
    (rotary_emb): LlamaRotaryEmbedding()
  )
  (lm_head): Linear(in_features=2048, out_features=128256, bias=False)
)
```

Let's dissect it.

First of all, we don't really know neither the order of operations or data type from this. But! [Model card](https://huggingface.co/meta-llama/Llama-3.2-1B-Instruct) on Hugging Face page tell us that weights are in [BF16](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format) format. We will go back to the format soon. 

We need to understand the order of operations to know how to code it. Sebastian Raschka has a gallery of LLM architectures and it shows nicely how the operations are organized - see [here (the left one)](https://magazine.sebastianraschka.com/i/168650848/61-qwen3-dense).

By looking at the diagram from Sebastian, we see that the operations order in LLama 3.2 1B is like this:

1. Send some text to the model
2. Turn it into tokens (a new concept, we didn't mention it yet)
3. Retrieve an embedding for each token
4. 16 transformer blocks, also called layers, which consist of:
  - RMS Norm
  - Residual connection
  - Masked grouped-query attention, which consists of:
    - Q projection
    - K projection
    - V projection
    - RoPE with Q projection
    - RoPE with K projection
    - Attention
    - Attention scores
    - Causal mask
    - Softmax
    - Residual connection
    - Attention scores with V projection
  - O projection (output projection)
  - Residual connection add
  - RMS Norm
  - [Feed forward](https://en.wikipedia.org/wiki/Feedforward_neural_network) (like in first neural networks, [Multilayer perceptron](https://en.wikipedia.org/wiki/Multilayer_perceptron)), which consists of:
    - Gate projection, first linear layer
    - Up projection, second linear layer
    - [SiLU](https://arxiv.org/pdf/1702.03118) activation function, similar to [ReLU](https://en.wikipedia.org/wiki/Rectified_linear_unit) but it's looking more like a sigmoid
    - Down projection, third linear layer
    - Residual connection add
5. RMS Norm
6. Linear output
7. Argmax

After these steps, we should get our first token produced by the language model ran on our server.

Don't worry if some of this operations are not familiar yet. You will understand them viscerally once we progress through the course. I keep forgetting how they work, too, and often need to look up again, so don't feel bad for going back and forth or looking things up with a chatbot or internet search.

## How floating-point numbers work and why we use bfloat16

Let's remind ourselves about what we want to achieve. We want to load a model. We already know the structure of the file with a model, a Safetensors file. We know our reference model architecture. We checked that model weights are stored in BF16 type. Let's take a moment to think about this type and floating-point numbers in general

Everything on the computer is binary, at the very end. [Computer numbers formats, too](https://en.wikipedia.org/wiki/Computer_number_format).

In ML models, weights are almost never integer numbers. They are rather real numbers. And computers are binary. It means that people had to figure out how to represent real numbers in programming languages, in a way that is memory efficient. In this context, it means that you can pack a lot of information in small amount of bits. Because of course, you can build a complex data type ("complex" like in "non-trivial", not like in "real+imaginary"), where the part to the left from a dot separator is stored as an integer and the part after a separator (so the less than 1 part) is stored as another integer. But you can see how inefficient is that. It's only useful for situations when you really need full precision. And this is why things like [decimal in Python](https://docs.python.org/3/library/decimal.html) exist.

Ok, so now let's think what could we do instead. The simplest alternative would be to take an integer number, like 1234 and say "I will put a dot between 12 and 34", so you'd get 12.34. People solved this with a scaling factor, like literally, for 1234 you'd use scaling factor 1/100 to get 12.34. You still need to figure out how to represent it on the computer. In this format, the fractional part (after the dot) has always fixed length - it's exactly in the middle. This format is called [fixed-point numbers](https://en.wikipedia.org/wiki/Fixed-point_arithmetic). It's less widely used than floating-point numbers

So if there are fixed-point numbers, there inevitably need to be non-fixed-point numbers. Let's just think about it, loosely. Non-fixed-point could mean that the dot (the point) can move around. The fractional part can be longer or shorter. Sounds like more memory efficient solution.

Floating-point numbers, like all other numeric types are represented as a sequences of bits. All of them have slightly different design choices, so let's focus on [regular 16-bit float](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#IEEE_754_half-precision_binary_floating-point_format:_binary16) (float16, FP16, IEEE 754-2008).

Float16 and many other floating-point numbers work in a similar way. They consists of three sections: sign, exponent and fraction. In total they take 16 bits.

```
[ sign | exponent  | fraction            ]
[ 0    | 0 1 0 0 1 | 0 0 1 0 0 0 0 0 0 0 ]
```

The intuition is like this: in floating-point numbers you can move the dot around. Fraction is the number you use to move the dot around. Like in the previous example, you fraction can be 1234. This time, dot is not fixed, so it can be anywhere you choose - 1.234, 0.0001234, 1234000 etc.

Sign is a single bit. 0 means the number is positive and 1 means negative.

Exponent is 5 bits. It controls the size of the number.

Fraction is 10 bits. It controls what number we use to move the dot around. Alternative names: significand, mantissa.

Now the formula: $(-1)^{sign} * 2^{exponent-bias} * (1.fraction)$

Notice two new things - the $1.$ before the fraction and $bias$. The `1.` is a design choice to increase the number's precision (you get 1 bit for free) without storing it explicitly in the memory. That's why it's called implicit. See this [nice Stack Overflow question and answer](https://stackoverflow.com/questions/4930269/floating-point-the-leading-1-is-implicit-in-the-significand-huh) for a better explanation. The main idea is that it's not some must-have thing, but rather a clever trick to store more data without using more memory, through saying in the floating-points numbers specification that the bit is always there (except for zero). The next new thing is $bias$ and look again where it is: $2^{exponent-bias}$. Do you have any hunches about why is it there? 

The answer is that it's there to be able to represent smaller than 1 numbers too. And specifically, to do it by making it possible to have a negative power of $2$. Exponent is always a positive number, represented as a few bits (5 bits, in float16). If we didn't subtract bias from it, $2^{exponent}$ would be always positive, and perhaps quite a big number (binary 11111 is 31 decimal). So we could generate any number bigger than 1 or smaller than -1 (almost any number, limited by the precision of floating-point number), but we couldn't represent any small but bigger than 0 number, like 0.0001234. Hence, we need subtract bias from the exponent. Bias can't be neither too big nor too small. Knowing that, what do you think the bias for float16 is? Remember that max value of exponent is 31

So the bias for float16 is 15. It means you can represent both smaller and bigger numbers.

There are more interesting things in specification of floating-point numbers, like how do we encode positive and negative infinites, how do we encode not-a-number value etc. [Wiki has many examples about exponent encoding](https://en.wikipedia.org/wiki/Bfloat16_floating-point_format#Exponent_encoding)

> The sizes of exponent and fraction are main differences between different types of floating-point numbers.

Let's take an example and encode number 12.34 as float16. 12.34 is positive, so sign bit is 0. To represent 12.34, we need to turn 12 into binary:

- 12 / 2 = 6, remainder 0
- 6 / 2 = 3, remainder 0
- 3 / 2 = 1, remainder 1
- 1 / 2 = 0, remainder 1

Read it backwards and you get 1100.

So decimal 12 is binary 1100. 

Now we turn the fraction 0.34 into binary:

- 0.34 * 2 = 0.68, integer part is 0
- 0.68 * 2 = 1.36, integer part is 1, we subtract 1 and continue
- 0.36 * 2 = 0.72, integer part is 0
- 0.72 * 2 = 1.44, integer part is 1, subtract 1 and continue
- 0.44 * 2 = 0.88, integer part is 0
- 0.88 * 2 = 1.76, integer part is 1, subtract 1 and continue
- 0.76 * 2 = 1.52, integer part is 1, subtract 1 and continue
- 0.52 * 2 = 1.04, integer part is 1, subtract 1 and continue
- 0.04 * 2 = 0.08, integer part is 0
- 0.08 * 2 = 0.16, integer part is 0

At this point, we have 10 bits computed, which is our length of fraction/mantissa/significand. At the first glance, it looks like we computed exactly as many bits as we need, but it's not entirely correct. It's because we need to include the first part of the number we encode - 12 (1100 in binary) - in the mantissa. $1.$ is always an implicit part of the number, so from $1100$ we remove the first $1$. We need to use what's left - $100$ - inside the mantissa. It means that we excessively computed 3 bits, which we will discard, because of 3 bits we need to use to include $12$ in the mantissa. So we didn't make a mistake, but we did some unnecessary work. Worth remembering if you decide to ever implement your own numeric type (and if you do it by yourself, [please let me know](jedrzej@maczan.pl)!)

Btw. look at the last term: "0.08 * 2 = 0.16, integer part is 0". The process is not finished, because we didn't reach 0 (we stopped at 0.16). It means that we will lose precision when representing this number.

Reading the bits from top to bottom, decimal 0.34 is binary approximately 0101011100.

Let's put 12 and 0.34 together and we get 1100 and 0101011100. Again, $1.0$ is implicit, so we're left with 100 and 0101011100. Put them together and you have 1000101011100. That's 13 bits. Remove 3 least significant bits (the ones to the right) and the final binary representation of fraction of our number is 1000101011.

> I know I'm sidetracking a bit, you were supposed to build an inference server. But honestly, why we need to rush? If these things are not interesting to you, then you can always skip forward. I believe there's some intristic joy of learning things in depth and if you're on the same page with me, let's continue

We're done with the mantissa. Look how did we turned $1100$ into $1.100$. If we were still in decimal world (base-10), we would say that to turn $1.100$ into $1100$ we need to multiply it by $1000$ ($10^3$). The numbers we operate on are binary (base-2), so it means that we need to multiply it by $2^3$, which immediately reminds us of $2^{exponent}$ from the floating-point formula. Which means that our exponent is $3$. But, the bias. Bias is 15. And in the formula for decoding the float, in the exponent of $2$ we subtract bias from the exponent that we read from floating-point bits (these 5 bits). Which means that we need to add the bias so it shows up in the stored exponent. So we add $3+15$ and our final exponent is $18$. Let's quickly turn 18 into binary format:

- 18 / 2 = 9, remainder 0
- 9 / 2 = 4, remainder 1
- 4 / 2 = 2, remainder 0
- 2 / 2 = 1, remainder 0
- 1 / 2 = 0, remainder 1

Read backwards and decimal 18 is 10010 binary.

We can put everything together to finally see decimal 12.34 as float16:

- sign bit: 0 (positive number)
- exponent: 10010 (18 binary, 3 because we shifted left by 3 positions + 15 bias; total 5 bits)
- fraction: 1000101011 (total 10 bits)

The number of bits correctly sums to 16. Together it looks like this:

```
[ sign | exponent  | fraction            ]
[ 0    | 1 0 0 1 0 | 1 0 0 0 1 0 1 0 1 1 ]
```

One interesting thing is that we know we lost some precision when turning 12.34 into a float. It was when we computed the 0.34 part. How about verify ourselves and decode our float back to decimal to see what value computer really stores when we try to store 12.34 as a float16? You curious, too?

Recall the formula: $(-1)^{sign} * 2^{exponent-bias} * (1.fraction)$

We know the number is positive, because sign is 0, so $-1^0 = 1$

We know the exponent is 18 minus bias 15 so it's 3. Plug it into the formula and we have: $2^3=8$

Fraction is 1000101011 and we need to turn it into decimal. We need to remember also about the implicit $1.$, so all together it's $1.1000101011$. Every next digit is multiplied by a power of 2, but with an exponent - 1 than the previous one. Similar to how it works when we turn a binary into decimal - we do the same but backwards, starting from the right we multiply each digit by 2 to the power of current position. Let's check it actually: exponent 10010 is then $0 * 1 + 1 * 2 + 0 * 4 + 0 * 8 + 1 * 16 = 18$, just like we thought. Ok, so now the mantissa. We turn $1000101011$ into decimal: $1 * 2^{-1} + 0 * 2^{-2} + 0 * 2^{-3} + 0 * 2^{-4} + 1 * 2^{-5} + 0 * 2^{-6} + 1 * 2^{-7} + 0 * 2^{-8} + 1 * 2^{-9} + 1 * 2^{-10} = 0.5 + 0 + 0 + 0 + 0.03125 + 0 + 0.0078125 + 0 + 0.001953125 + 0.0009765625 = 0.5419921875$. So putting it together with implicit $1.$, our number is $1.5419921875$.

Let's plug all of them into the formula together: $1 * 8 * 1.5419921875 = 12.3359375$. Correct and close enough, at least in LLM inference. The error: $12.34 - 12.3359375 = 0.0040625$

But this was float16. And our reference model (Llama 3.2 1B Instruct) used bfloat16 (BF16). It turns out bfloat16 is widely used in many different models. It also takes 16 bits, but it's exponent is longer (8 bits) than regular float16 (5 bits). First interesting thing about bfloat16 is that the size of exponent is actually the same as of twice bigger 32-bit float (float32, FP32, float), which also has 8 bit exponent. So comparing with float16, bfloat16 trades 3 bits of fraction to gain 3 bits of exponent. So naturally, the fraction shrinks and it's only 7 bits now. So to sum up - bfloat16 is 1 sign bit, 8 bit sign exponent and 7 bit fraction.

Why bfloat16 matters? It has the same size as float16 (a half size of a full precision float), and at the same time it has the same exponent as the float32 - at the cost of smaller fraction. The industry often chooses it for inference, because it's less likely to have range issues ([overflow](https://en.wikipedia.org/wiki/Integer_overflow) / [underflow](https://en.wikipedia.org/wiki/Arithmetic_underflow)) and at the same time the loss of precision (due to smaller fraction) is an acceptable tradeoff in LLM inference (empirical results shows it).

If you want to test your understanding, Wikipedia page about half-precision floating-point formats has [lots of good examples](https://en.wikipedia.org/wiki/Half-precision_floating-point_format#Half_precision_examples)

Now you have all prerequisities to load a model in your inference engine and start doing interesing things. Would like to read some hands-on primer on working with GPU memory in CUDA? Keep reading the next section or skip to the inference part.

Oh btw, 32-bit floats are called single-precision floats. And then you hear about double-precision. And you might be like - omg so I get TWO floating points within the same number, then? Unfortunately the double precision just means that the number is bigger (64-bit). Meh

## GPU and CPU memory

This chapter might be a bit random, but I hope it's useful if you still begin with GPU programming. Data can live in host or device. 

Host is your PC with CPU. It has slow, big memory - DRAM - the one you buy and plugin into a motherboard. Recently very pricey, because of big tech AI craze. DRAM is separate from CPU. CPU has also it's own, small and fast on-chip memory - SRAM (I will wait until you stop laughing, if you're Slavic). 

Device is your GPU. It also has slow, big memory - HBM, VRAM - but it's soldered to the card and you can extend it easily, as you do with DRAM in your PC. It has small, fast memory - SRAM - which you can use if you defined `__shared__` variables.

GPU can't access your DRAM, so before running any computation on GPU, you need to copy data to it. The typical flow can be like this:

1. Create a variable on CPU
2. Write to variable on CPU
3. Compute the size of memory taken by the variable on CPU, or figure the max size some other way (sometimes you know the maximum number of elements in vector or alike)
4. Multiply the computed size by the size of variable type
5. Allocate the memory on GPU
6. Copy the variable from CPU to GPU
7. You can use this data in your GPU computations now

Ideally you want to allocate as little memory as possible, reuse this memory as much as possible and copy data as rarely as possible.

Let's have an example: I need to know what tokens are currently active during the inference. For this purpose, I need this data both on CPU and GPU.

1. I create a vector on CPU:
```cpp
std::vector<int> active_tokens;
```
2. I write to the vector some tokens, like
```cpp
active_tokens.push_back(token);
```
3. Now I need to figure out how many active_tokens there might ever be (max number), to allocate the GPU memory accordingly. In this case, I know that it might be up to BATCH_SIZE (say, max 2 elements always)
4. Tokens are ints, so the final size of memory I need to allocate is
```cpp
BATCH_SIZE * sizeof(int)
```
5. Declare and allocate the GPU memory
```cpp
int *gpu_active_tokens;
cudaMalloc(&gpu_active_tokens, BATCH_SIZE * sizeof(int));
```

`cudaMalloc` is the CUDA function you use for allocating GPU memory. You need a pointer on host - `int *gpu_active_tokens` - to be able to use this memory and pass it to your GPU kernels.

6. Copy data from CPU to GPU. We know the maximum amount of active tokens (BATCH_SIZE), but in the case if it's currently less active tokens processed, we need to compute amount of them: `num_active_slots * sizeof(int)`

```cpp
//               destination,               source,           size of data to copy,   direction of copying
cudaMemcpy(gpu_active_tokens, active_tokens.data(), num_active_slots * sizeof(int), cudaMemcpyHostToDevice);
```

7. Now you can use this data when you invoke CUDA kernels, like
```cpp
embeddingGatherDecode(gpu_active_tokens, num_active_slots, hidden_state, weights.embed_tokens);
```

## Single token inference

We start working on inference now. You can either close this course now and start coding, based on the operations sequence we laid out in [Safetensors and your model section](#safetensors-and-your-model) or you can keep reading to fill your brain cache with useful stuff based on which you will have an easier time implementing the model. Your choice 

Regardless of what you choose, I'd like to share with you what is like to me to learn new things using LLMs as of 2026, maybe you'll find it useful when working on this course. I think about the learning process as of this loop:

1. Understand more-or-less what you want to build
2. Describe your mental model to a chatbot, ask it to figure out your blind spots, mistakes in your thinking, fill gaps in your knowledge with tailored information and enough context to understand
3. Read the answer, try to interalize it and update your mental model
4. Repeat until no mistakes
5. Start coding, continue until blocked
6. When blocked, go back to 2.

One thing that I find most important is to remember that a human learns by putting a real effort into understanding. Figuring things out, making mistakes, debugging our thought processes, thinking about our thinking, updating our understanding of things by describing them to others and recognizing the spots where our understanding is insufficient or wrong. All these things we need to do intentionally, because we can always delegate it to LLM. I am strong proponent of LLMs as a personalized tutor and it's a best thing to learn with, as of 2026. Also, I think there are 2 different types of effort and the confusion about whether and how to learn with LLMs arises. One type of effort is the effort of understanding, where you rewire your brain by thinking about a thing you want to understand. You break it down into smaller pieces, understand the mechanism, the data flow, the relationships, you relate it to other things you know, you imagine, you recreate, you make mistakes and they point you where you should look next. Digging deep into a concept you want to learn is one of the highest beauties of life. This type of effort and struggle is where learning happens. And then there's also a waste energy effort. Things like - data retrieval, commute, mechanistic actions that eat up your time, which you are able to fully do by yourself, you understand them from the inside out, and they learn you anything new anymore - this is a type of effort I think has too low ROI to perform, and it's better to delegate it to LLM. You can argue that in some sense it's still delegating your thinking to LLM and you're not wrong. You can always get better at reading documentation (you will do that anyway when learning, but often you can skip this part for one-off things) or internet search. But your time, focus and energy is limited. If becoming marginally better at internet search comes at the expense of learning more about the current topic you learn - then it's not a good tradeoff. YMMV though, not an expert, I'm just an internet random

You have a comfort of always being able to take a look at my code if you want to understand how I built it. Most of the time, you probably don't need to. But if you're stuck or when you finish a certain thing and want to compare - then ofc feel encouraged to do that. 

I'll lead you through a blueprint of a program - from an empty page to single token inference. You have now a lot of info you needed to start. 

So, you need to bootstrap the project. Import CUDA runtime, cuBLAS and `nlohmann::json`. This is the kind of work you can freely copy-paste from my code, if you don't want to do it.

```cpp
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <queue>
#define JSON_USE_IMPLICIT_CONVERSIONS 0 // this one is useful, but I don't remember why
#include "json.hpp"

using json = nlohmann::json;
```

If you're using VS Code and its complaining about unresolved imports, you need to adjust paths in `.vscode/` files. 

A useful helper function that you could run once at the beginning of your `main()` function is some debug info about the GPU your CUDA chose:

```cpp
int checkGPUStatus()
{
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    if (device_count == 0)
    {
        std::cerr << "No CUDA devices found\n";
        return 1;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Device: " << prop.name << "\n";
    std::cout << "Compute capability: " << prop.major << "." << prop.minor << "\n";
    std::cout << "Global memory: " << prop.totalGlobalMem / B_TO_MB << " MB\n";
    std::cout << "SM count: " << prop.multiProcessorCount << "\n";
    std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
    size_t free_mem;
    size_t total_mem;
    cudaMemGetInfo(&free_mem, &total_mem);
    std::cout << "Free memory: " << free_mem / B_TO_GB << "GB, total memory: " << total_mem / B_TO_GB << "GB\n";
    return 0;
}
```

Ok, now load the model. You know the safetensors file structure. There are many possible approaches to how you will store the model. The approach I like and I think you might like as well is to load the tensors as a single big block to the GPU memory. You need to allocate this memory first and safetensors file header will tell you how much. Then, using offsets in the header you will map the regions in GPU memory into pointers on CPU, so you know where to look when you want to, say, retrieve weights of K in layer 5. Some weights are not layer-specific, make sure to recognize that. One tip: make yourself a favor and print a lot. Debuggers are great, but not that much useful for working with raw data. Write (or generate by LLM) helpers scripts in Python to print dumps of models. I do lots of such things and they help me understand the data, model, etc. Anything that helps you move forward and strenghtens your understanding is ok, if you ask me. 

Oh btw, the data type of model is bfloat. In CUDA, it's `__nv_bfloat16`

If you implement weights in a similar way to what I described above, then to retrieve weights of K in layer, you'll write something along these lines:

```cpp
weights.w_k[5] = (__nv_bfloat16 *)((char *)model_weights + offsets.at("model.layers.5.self_attn.k_proj.weight"));
```

You need to allocate data for all tensors your model uses. Don't be afraid to overallocate and make it suboptimal. The first goal is to make it work.

In section [Safetensors and your model](#safetensors-and-your-model) we laid out the exact sequence of operations we need to implement to predict the first token. Once we make it work and actually get our first token generated, we will reuse a lot of the code and write a loop around it.

## Tokenization

Ok, so I assume you loaded the model and mapped the weights of the model to some useful pointers. Now we need to read user's input, the prompt, and turn it from a text to something that model understands - tokens. All mainstream LLMs use tokens, not words or characters.

To turn text into a sequence of tokens, you need a tokenizer. We will use an existing tokenizer, which produces tokens that match the dictionary of Llama 3.2 1B. See file [python/tokenizer.py](python/tokenizer.py), where I use a tokenizer from Hugging Face

Going deep into tokenizers is out of the scope, what you really need to remember is that it takes a text and produces a sequence of tokens (ints), which represent your text but as a vector of ints. And LLM needs your text as this vector of ints.

> Building your own tokenizer is quite a fun thing. I wrote mine 3 years ago and feel free to use it as a reference, if you'd like to learn more about tokenizers: https://github.com/jmaczan/bpe-tokenizer. There's also a great resource from Andrej Karpathy where he builds a tokenizer, and it's very useful and educational: video https://www.youtube.com/watch?v=zduSFxRajkE, code https://github.com/karpathy/minbpe and this article https://github.com/karpathy/minbpe/blob/master/lecture.md

## Embeddings

Your text is translated into tokens and you feed the tokens into your LLM inference server. Tokens are more like indices, but they are not the data on which your LLM is going to work on. Large language models know how to map each token to a vector, where every token has the same vector length, but different vector values. These vectors are called embeddings. They embed the meaning of a token. Then you feed a list of tokens into the LLM, it retrieves one embedding per token, where tokens work as indexes that tell the model which embedding (vector) it should retrieve from it's weights. In our case, every embeddings has 2048 length. So for 5 input tokens, you get 5 vectors of 2048 length, which together is a matrix of dimension (5, 2048). We already know the type of every number in these embedding vectors - its bfloat16.

This might be a first CUDA kernel to write in this course. Your job is to retrieve the embeddings for all the input tokens.

What do you need to do that? You need input tokens and embeddings weights. You already loaded embeddings weights on your GPU. But input tokens you provide live on CPU - you can provide them as CLI params, hardcode or read from a file. By default, you load them into some vector of ints, probably. Now you need to make them available to your GPU. So what you do now?

You can create a buffer on GPU where you copy your input tokens. When you do it, you will be able to use the pointer to these input tokens on GPU and pass the pointer to your CUDA kernels. This time, I will help you do it. Any other kernels and CUDA data move/allocation you will write on your own, unless they will be exceptionally interesting, okay?

Let's say you have your input tokens in a vector of ints on CPU:

```cpp
std::vector<int> input_tokens = {678, 264, 1933, 13};
```

Your model, like all models, has a constraint about how many tokens it can process. For Llama 3.2 1B it's 2048. It includes both input and output tokens. We need to arbitrarily choose how many of them we allow to be the prompt size. Let's say it's going to be max 512 tokens as an input and we declare it as a [constexpr](https://en.cppreference.com/w/cpp/language/constexpr.html):

```cpp
constexpr int MAX_PROMPT_LEN = 512;
```

You need a copy of your input tokens on GPU. So, you need to allocate a memory on GPU and you need a pointer to this memory. A pointer:

```cpp
int *gpu_input_tokens;
```

To allocate the memory on GPU, we will use `cudaMalloc` function, which you already know from a chapter about GPU memory. The first argument of `cudaMalloc` is a pointer to our pointer (`void **`). The second argument is a size of memory to allocate. We know how many tokens we can maximum have as an input. The tokens are integers, so the size of memory to allocate is max number of input tokens * size of an int.

```cpp
cudaMalloc(&gpu_input_tokens, MAX_PROMPT_LEN * sizeof(int));
```

Ready to copy input tokens into GPU now:

```cpp
//              destination,              source,                              size, a direction of copying
cudaMemcpy(gpu_input_tokens, input_tokens.data(), input_tokens.size() * sizeof(int), cudaMemcpyHostToDevice);
```

We can write a CUDA kernel now. 

## CUDA kernel engineering - embeddings

> There exist much better resources than I can produce, so to learn CUDA, please check out [CUDA Programming Guide](https://docs.nvidia.com/cuda/cuda-programming-guide/01-introduction/introduction.html). I'll just briefly mention a basics here, but they might not be sufficient for you

Kernels are functions that are executed on a GPU. You launch the same function multiple times. Every launched function is a separate thread. They run the same code. They receive slightly different parameters, like index of a thread. Threads are grouped into blocks. When you launch a CUDA kernel, you define how many blocks you want to invoke and how many threads are there in every block. Threads are also grouped into warps. Every warp has 32 threads. So, when you run your kernel and you define that it has to run 5 blocks and each block has to run 64 threads, then it means that each block runs 2 warps, 32 threads in each warp. 

When writing CUDA kernels, a lot of effort goes into thinking about memory. I mean, it's like thinking from the thread perspective: "what data should I process?", "where I should write the results?". It in practice means figuring out index of input data you need to read and where you should write the output to. Your main tools are built-in variables, like threadIdx, blockIdx and blockDim - see this link for more structured info https://docs.nvidia.com/cuda/cuda-programming-guide/02-basics/writing-cuda-kernels.html. Every thread has it's own values of these variables. This makes possible to running the same computation in parallel. This approach is called [SIMT](https://en.wikipedia.org/wiki/Single_instruction,_multiple_threads).

Ok, so what we know is that for every input token, we want to retrieve an embedding which consists of 2048 bfloat16 numbers. The first approach that we can think about is - okay, so we have N tokens to retrieve, and each token needs to retrieve 2048 numbers. So we can run N blocks - one block per token - and 2048 threads in every block, so every thread would retrieve exactly a single number. Let's write an empty kernel and write down how we would like to execute it with N blocks and 2048 threads per block.

Empty kernel:
```cpp
__global__ void embeddingGatherKernel()
{
}
```

An invocation of this kernel:
```cpp
embeddingGatherKernel<<<num_input_tokens, 2048>>>();
```

Looks good. Now let's think what data we need on the input and what we want to produce. We need input tokens and weights of embeddings of the loaded model. We also need some output to write to. We already copied tokens to GPU and we have weights of embeddings on GPU, too. So the output, what it should be like? We retrieve 2048 bfloat16 numbers for N tokens. So we need to have a memory, to which we can write that many numbers. Let's allocate it first:

```cpp
__nv_bfloat16* input_embeddings;
cudaMalloc(&input_embeddings, MAX_PROMPT_LEN * sizeof(__nv_bfloat16) * 2048);
```

Size of memory we allocate: max number of input tokens (MAX_PROMPT_LEN) * 2048 * size of every number (sizeof(__nv_bfloat16)).

So let's pass these pointers into our kernel. `weights.embed_tokens` are weights of embeddings we have loaded from the safetensors file:

```cpp
embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, input_embeddings, weights.embed_tokens);
```

Let's update the kernel signature to accepth these arguments:
```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
```

Okay. So now we need to actually retrieve the numbers. Remember, we think about the kernel from the perspective of a single thread, where there are multiple threads invoked with exactly the same code, but their values of `threadIdx.x`, `blockIdx.x` (and a few similar variables) are different - that's how they know which thread they are and what data they need to take and what data they need to produce. 

Let's compute the index of output first. We know it's of dimension (number of tokens, 2048). Let's assume it's row-major format, so it's easier to work with it. In other words, it means when you access index 0, you get 0th token of 0th number, index 1, you get 1st token of 0th number, and like this up to 2047 index. Then, it starts again with 0th token of 1st number. `threadIdx.x` is unique within a block. So, if we had only a single token, we could map 1:1 `threadIdx.x` to `input_embeddings` index (the output memory), like input_embeddings[threadIdx.x]. For a moment, let's assume this is the case - that we always get a single token. So, we now know where we write to. We need to retrieve the correct number of an embedding for this token. Every embedding is 2048 numbers, so to get to the beginning of the embedding of id of the token, we need to multiply the token by 2048 - `gpu_input_tokens[0] * 2048` (again, assuming `gpu_input_tokens` is a single token, so that's why we can access the first element of it). This way we receive the index of the first number of the token's embedding. The embedding has 2048 numbers and each thread in a block retrieves a single number of this embedding. So, we need to move the index by the current value of `threadIdx.x`, which will tell us which number of the embedding we need to retrieve. So, the index becomes `gpu_input_tokens[0] * 2048 + threadIdx.x`. Let's combine it into a kernel implementation:

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    input_embeddings[threadIdx.x] = embed_tokens[gpu_input_tokens[0] * 2048 + threadIdx.x];
}
```

Looks good. Let's make it work for more than a input single token. What changes? We wanted to run multiple blocks - one block per input token. So we need to take this into account in index arithmetic. How will current thread's output index look like? Previously, it was just `threadIdx.x`. Now we need to take into account that there are multiple tokens. Every token results with 2048 sized embedding. So, we can multiply block's index by 2048 and add it to `threadIdx.x` to get a position in an embedding for currently processed token (and token idx == block idx). So it becomes:

```cpp
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

We can't hardcode first token anymore `gpu_input_tokens[0]`. Since token idx == block idx, we can replace `[0]` with `blockIdx.x`:

```cpp
embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
```

Combining it together and we get:

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
}
```

We create a function that is a wrapper over this kernel invocation, so you can run it from your C++ code. Kernels are in `.cu` files, because of their specific syntax, like the `<<<>>>` where you define the grid (blocks and threads). 

```cpp
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens)
{
    embeddingGatherKernel<<<???, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Wait a sec. We don't know a number of tokens. We don't use C++ data structures anymore, like vectors, so we can't read the number of elements of `gpu_input_tokens`. We need to pass it to this function explicitly:

```cpp
void embeddingGather(int *gpu_input_tokens, __nv_bfloat16 *gpu_input_embeds, __nv_bfloat16 *embed_tokens, int num_input_tokens)
{
    embeddingGatherKernel<<<num_input_tokens, 2048>>>(gpu_input_tokens, gpu_input_embeds, embed_tokens);
}
```

Everything looks good now! Let's launch it. I encourage you to actually do it.

And then?

...

[It's a trap!](https://www.youtube.com/watch?v=4F4qzPbcFiA) Max threads per block is 1024, at least for most of NVIDIA GPUs you might have at home. In `checkGPUStatus()` function we event print this info: 

```cpp
std::cout << "Max threads per block: " << prop.maxThreadsPerBlock << std::endl;
```

So, what we can do now? We can either launch 2 times more blocks or process 2 numbers in every thread instead of just one number. I go with the second option - it will be probably faster, doesn't require launching more threads and doesn't require any synchronization between threads or even between writing the output memory.

We have max 1024 threads per block. Embedding has 2048 numbers. We want to process two numbers from embedding per thread. What options do we have? We can either process current thread's number and the number next to it or we can process current thread's number and the number on the same position in the next half of the embedding, so current thread's index + 1024. Would the first approach work? First thread would process 0th and 1st number. Second thread would process 2nd and 3rd. Third 4th and 5th. Probably would work too (?), but more index arithmentic. The second option is easier again. Just add 1024 to both input and output indexes.

```cpp
__global__ void embeddingGatherKernel(int *gpu_input_tokens, __nv_bfloat16 *input_embeddings, __nv_bfloat16 *embed_tokens)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input_embeddings[workIndex] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x];
    input_embeddings[workIndex + 1024] = embed_tokens[gpu_input_tokens[blockIdx.x] * 2048 + threadIdx.x + 1024];
}
```

Run it. This time it will work.

Congrats to you, you finished your first CUDA kernel!

Let's move back from computation and low-level programming to semantics/meaning of what we just did. Notice that while all these embeddings we retrieve for input tokens have some encoded meaning within them, they don't know about each other. They don't know their position within a text we provided as an input. They don't know what tokens they are surrounded with. They don't know the conversation history, etc etc. This understanding will be build and stored as K and V projections.

## RMSNorm and parallel reduction in CUDA

Look back at the sequence of operations in our model (section [Safetensors and your model](#safetensors-and-your-model)). After we retrieve the embeddings for our tokens, it's time for [RMSNorm](https://arxiv.org/abs/1910.07467). Unlike embeddings gather, it's a first operation that will run in layers. Our model, Llama 3.2 1B, has 16 layers. RMSNorm takes our retrieved embeddings and - using model weights for a rms norm `weights.input_layernorm[layer]` - runs RMSNorm function. RMSNorm is an operation that modifies all numbers in an embedding. To do that, first it needs to see all the elements and compute their [root mean square](https://en.wikipedia.org/wiki/Root_mean_square) sum.

Based on the paper, the formula is:

$$ \text{normalized}_i = \frac{a_i}{\text{RMS(a)}} \text{, where RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i} $$

Let's write this kernel together. 

To compute RMSNorm, we need to compute `RMS(a)` first. It requires going through all the numbers, so we will need some synchronization between threads. We normalize each embedding separately. So again, we launch as many blocks as there are input tokens. Embeddings have 2048 numbers, but 1024 is max threads per block, so we will use the same trick as we used in embeddings gather kernel - we will process two numbers within each thread. We need some temporary value, to which we can write the squares of every number of an embedding. We could use a single variable for it. Every thread within a block would write a square of its number to it. The problem here is synchronization of reads and writes to the shared variable - preventing race condition, ensuring that we will end up with a correct sum. There is another approach, where we create a temporary vector of the same length as number of threads in a block, and each thread writes a square of its value to `threadIdx.x` position in a temporary vector. This way, we ensure that no two threads will try to read or write to the same position in the vector. This technique is called [parallel reduction](https://developer.download.nvidia.com/assets/cuda/files/reduction.pdf) (tree reduction).

Let's create a vector. We use `__shared__` keyword to indicate that the vector is shared among all threads of a block.

One thing about numerical stability before we move on - remember that our data is bfloat16 format. Main strengths of bfloat16 is a big exponent - 8 bits, same size as float (float32) - but the mantissa is small (7 bits only vs 10 bits in float16). It means that to not lose a precision in our computation, we can use a bigger type to actually compute things inside the kernel - things like a square of each number. We'll use float for this purpose. We will cast every number to float and declare our temporary buffer as float, too.

```cpp
__shared__ float rms_vector[1024];
rms_vector[threadIdx.x] = (float)input[threadIdx.x] * (float)input[threadIdx.x];
```

We wanted to compute two numbers per thread, so we either make `rms_vector` bigger (2048 items) or append the square of a number 1024 indices away to our `rms_vector[threadIdx.x]`. The second one is easier to implement and will be probably much faster, because we will have less steps in a tree reduction - it's faster to reduce to 1 element from 1024 elements than from 2048 elements.

```cpp
__shared__ float rms_vector[1024];
rms_vector[threadIdx.x] = (float)input[threadIdx.x] * (float)input[threadIdx.x] + (float)input[threadIdx.x + 1024] * (float)input[threadIdx.x + 1024];
```

It would work, if we launched just a single block. But again, we want to be able to process multiple tokens, so we will launch as many blocks as tokens. Because of that, we need to figure out the correct item indexes for current thread. `threadIdx.x` is correct as a index of `rms_vector`, because `rms_vector` is always 1024 floats. `threadIdx.x` won't do as an input index, because we have multiple tokens on the input. We know each tokens takes 2048 `__nv_bfloat16`s of space. An index of a block tells us an index of a token. So, to move to the current token in the input, we need to multiply `blockIdx.x` by size of a token - by 2048. Our input index for a current thread is then:

```cpp
int workIndex = threadIdx.x + blockIdx.x * 2048;
```

Now, the reduction part. The idea is that every i-th element adds an element at the index of `self + i` to itself. And then, we multiply i by 2 and repeat. After every iteration, we need to make sure that all the threads finish writing to the `rms_vector` before we move to the next iteration. CUDA has [`__syncthreads()`](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/#thread_synchronization), which we can use for this purpose, and we put it after we write to `rms_vector`.. It can be done in a loop, but we will start with writing out everything explicitly, so the tree reduction algo will feel more understandable. The algorithm finished when we get to the `self + 1024`. The sum of all elements is stored at 0 index - `rms_vector[0]`. Let's write down the code so you can actually see it yourself. To me, many pieces weren't obvious until I wrote it down myself. Maybe you'd like to try it to, before reading the code? 

Please do whatever you feel more comfortable with, and I will write down the verbose (but correct) version of the RMSNorm kernel anyway:

```cpp
__shared__ float rms_vector[1024];
int workIndex = threadIdx.x + blockIdx.x * 2048;
rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
__syncthreads();
```

Each thread in a block stores its input number's square summed with a square of the input number, shifted right by 1024 (covering the second part of 2048 vector, which we can't cover by launching 2048 threads in a block, because of CUDA limitation to max 1024 threads per block). `rms_vector[0]` contains 2 squares out of 2048 numbers of a currently processed input embedding (for a current token) - indices `0` and `1024`.

We can go to the first step of reduction. Every second thread will add the next number in the `rms_vector`. Remember that every number in rms_vector currently already stores two squares.

```cpp
if (threadIdx.x % 2 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1]; 
}
__syncthreads();
```

`%` is modulo operator ([division with remainder](https://en.wikipedia.org/wiki/Euclidean_division)). Now `rms_vector[0]` stores squares of numbers on indices `0`, `1024` and `1`. We know that `1` already contained `1` and `1025`, so `rms_vector[0]` has actually these 4 squares now: `0`, `1`, `1024` and `1025`.

We increase the "gap" by 2, so now every 4-th element will add an element that is 2 indices away from self:

```cpp
if (threadIdx.x % 4 == 0) {
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2]; 
}
__syncthreads();
```

Notice that the index we add to the `threadIdx.x` is half of what we divide with (we divide with reminder by 4 and add the index that is 2 items away). Think about why do we do this and if the summation would give the correct result when we would add the same number by which we modulo.

The process continues until we modulo `threadIdx.x` by 1024:

```cpp
if (threadIdx.x % 2 == 0)
{
    // every second item has its predecessor
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 1];
}
__syncthreads();
if (threadIdx.x % 4 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 2];
}
__syncthreads();
if (threadIdx.x % 8 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 4];
}
__syncthreads();
if (threadIdx.x % 16 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 8];
}
__syncthreads();
if (threadIdx.x % 32 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 16];
}
__syncthreads();
if (threadIdx.x % 64 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 32];
}
__syncthreads();
if (threadIdx.x % 128 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 64];
}
__syncthreads();
if (threadIdx.x % 256 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 128];
}
__syncthreads();
if (threadIdx.x % 512 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 256];
}
__syncthreads();
if (threadIdx.x % 1024 == 0)
{
    rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + 512];
}
__syncthreads();
```

Now `rms_vector[0]` holds the sum of all the squares of its token's embedding numbers. To compute $\text{RMS(a)}$ we need to divide it by a size of an embedding (2048) and take a square root of it: $\text{RMS(a)}=\sqrt{\frac{1}{n}\sum_{i=1}^{n}a^2_i}$.

```cpp
if (threadIdx.x == 0)
{
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0); // = RMS(a)
}
__syncthreads();
```

Now all threads can read `rms_vector[0]` to retrieve $\text{RMS(a)}$. Let's finish our kernel with computing a normalized value for current thread's numbers (the one that corresponds to threadIdx.x and threadIdx.x + 1024). Remember that we want to use `float` when doing math that might be affected by limited precision of `bfloat16`. So, we cast all numbers to `float`s and, before writing to the output memory, we cast it back to `__nv_bfloat16`. 

```cpp
output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);

output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
```

Almost done!

There is a problem with these two lines. Let's recall at how the $\text{RMS(a)}$ is used when computing a normalized value: $\text{normalized}_i = \frac{a_i}{\text{RMS(a)}}$. If $\text{RMS(a)}$ is 0, then this division becomes a division by 0 and in C++ we'll get NaN or infinity, both means that we lost and even one NaN will corrupt the inference and our engine will produce a garbage or crash. To prevent that from happening, the refernce LLM we use has an epsilon parameter, which defines what value we add to the result of RMSNorm. 

```py
(input_layernorm): LlamaRMSNorm((2048,), eps=1e-05)
```

This way, we prevent division by zero and NaNs propagation from happening. Let's just update a line where we compute $\text{RMS(a)}:

```cpp
if (threadIdx.x == 0)
{
    rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5); // = RMS(a)
}
__syncthreads();
```

Congrats on finishing your RMSNorm CUDA kernel! For a verbosity, we can replace the manual writing down every iteration of the reduction with a for loop and we're done here

```cpp
__global__ void rmsNormKernel(__nv_bfloat16 *input, __nv_bfloat16 *output, __nv_bfloat16 *norm_weights)
{
    __shared__ float rms_vector[1024];
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    rms_vector[threadIdx.x] = (float)input[workIndex] * (float)input[workIndex] + (float)input[workIndex + 1024] * (float)input[workIndex + 1024];
    __syncthreads();
    // tree reduction
    for (int i = 1; i < 1024; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0)
        {
            rms_vector[threadIdx.x] = rms_vector[threadIdx.x] + rms_vector[threadIdx.x + i];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        rms_vector[0] = sqrt(rms_vector[0] / 2048.0 + 1.0e-5);
    }
    __syncthreads();

    output[workIndex] = (__nv_bfloat16)(((float)input[workIndex] / rms_vector[0]) * (float)norm_weights[threadIdx.x]);
    output[workIndex + 1024] = (__nv_bfloat16)(((float)input[workIndex + 1024] / rms_vector[0]) * (float)norm_weights[threadIdx.x + 1024]);
}
```

## RoPE

In our reference model, the next operation after RMSNorm is [RoPE](https://arxiv.org/pdf/2104.09864), a way of encoding tokens position into the hidden state (embedding). Very approachable description of positional encoding using RoPE is [here by Christopher Fleetwood](https://fleetwood.dev/posts/you-could-have-designed-SOTA-positional-encoding).

Try to write it on your own, if you have an energy for that. If not, here's my finished kernel. There are some things that I could optimize, like some values can be precomputed once and then shared - see theta and angles

```cpp
__global__ void ropeKernel(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    if (2 * threadIdx.x + 1 + blockIdx.x * proj_dim < num_tokens * proj_dim)
    {
        // TODO: precompute thetas, angles and perhaps sin/cos vals and reuse it across all kernel invocations
        int double_i = 2 * (threadIdx.x % 32);
        float theta = 1.0 / (pow(500000.0, ((float)double_i / HEAD_DIM)));
        float angle = blockIdx.x * theta;
        __nv_bfloat16 prev_2i = input[2 * threadIdx.x + blockIdx.x * proj_dim];
        __nv_bfloat16 prev_2i_1 = input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim];
        input[2 * threadIdx.x + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * cos(angle) - (float)prev_2i_1 * sin(angle));
        input[2 * threadIdx.x + 1 + blockIdx.x * proj_dim] = (__nv_bfloat16)((float)prev_2i * sin(angle) + (float)prev_2i_1 * cos(angle));
    }
}

// proj_dim: q_proj 2048, k_proj 512
// num_threads: I want to use it for both q_proj and k_proj so need to parameterize num_threads (1024 for q_proj and 512 for k_proj)
void rope(__nv_bfloat16 *input, int num_tokens, int proj_dim)
{
    int num_threads = proj_dim / 2;
    if (num_threads > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, RoPE kernel not launched";
        return;
    }

    ropeKernel<<<num_tokens, num_threads>>>(input, num_tokens, proj_dim);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

< TODO describe in more details >

## Residual connections

It's a simple technique used in many different deep learning models, where you add inputs to the outputs, so your input data is never fully gone. It's done for stability in training. See [more info here](https://towardsdatascience.com/what-is-residual-connection-efb07cab0d55/). From our perspective it's just adding two same sized vectors, elementwise.

```cpp
__global__ void residualKernel(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds)
{
    int workIndex = threadIdx.x + blockIdx.x * 2048;
    input[workIndex] = input[workIndex] + input_embeds[workIndex];
    input[workIndex + 1024] = input[workIndex + 1024] + input_embeds[workIndex + 1024];
}

// (num_tok, 2048) + (num_tok, 2048) -> (num_tok, 2048)
void residualAdd(__nv_bfloat16 *input, __nv_bfloat16 *input_embeds, int num_tokens)
{
    residualKernel<<<num_tokens, 1024>>>(input, input_embeds);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

## cublasGemmEx

[Matrix multiplication](https://en.wikipedia.org/wiki/Matrix_multiplication) is one of the main operations used in deep learning, most notably in large language models. Matrix is a table of numbers. It has rows and columns. A single row and a single column is called a vector -- a sequence of numbers. Matrix multiplication uses two matrices, A and B, as an input and produces matrix C as an output. Matrix A has dimensions (M, K). Matrix B has dimensions (K, N). Both matrix A and B share the same dimension K. In other words, rows of matrix A have the same length as columns of matrix B. When you multiply A by B, you get a new matrix C with dimensions (M, N):

$$A (M,K) \times B(K,N)=C(M,N)$$

Every element of matrix C ($c_{ij}$) is a [dot product](https://en.wikipedia.org/wiki/Dot_product) of i-th row of A and j-th column of B. Dot product is a sum of all pairs, where each pair is a result of multiplying numbers from i-th row of A with numbers from j-th row of B on the same indices within their vectors (both row and column are vectors):

$$c_{ij} = a_{i0}b_{0j} + a_{i1}b_{1j}+...+a_{ik}b_{kj}=\sum_{x=0}^{k} a_{ix}b_{xj}$$

Back to large language models. Matrix multiplication happens when computing attention and Q, K and V projections. Most popular hardware to compute it efficiently are NVIDIA GPUs. They provide an important library, [cuBLAS](https://developer.nvidia.com/cublas), which allows you to run high performance linear algebra computations on their GPUs, including matrix multiplication, using [cublasGemmEx](https://docs.nvidia.com/cuda/cublas/index.html#cublasgemmex) function.

## The column-major to row-major transposition trick

**TL;DR: if your data is in row-major format and you're going to use cuBLAS, then set transposition flag to `CUBLAS_OP_T` for matrices that are not tranposed yet, and `CUBLAS_OP_N` to matrices that are transposed in your formula**

Now the derivation and understanding:

The problem with cublasGemmEx is that it expects you to provide the matrices in [column-major format](https://en.wikipedia.org/wiki/Row-_and_column-major_order). And the LLMs, like Llama 3.2 1B Instruct, are distributed in row-major format.

![column and row major diagram](assets/column-row-major.png)

It turns out we don't have to modify the data format to use cuBLAS matrix multiplication functions. All thanks to these properties:

$$[A^T]_{ij}=[A]_{ji} \qquad C^T=B^T \times A^T \qquad (A^T)^T=A$$

The "$^T$" means that we transpose the matrix. Transposing a matrix turns columns into rows, and rows into columns. When you store the matrix in row-major format, and cuBLAS reads it in column-major format, it's an equivalent of transposing the matrix.

Let's see an example to understand it better: we want to compute $C = A \times B$, where A has dimensions (5, 2048) and B has dimensions (512, 2048). Our desired dimension of C is (5, 512). Right now, A and B dimensions are incompatible: $A(5, 2048)$ and $B(512, 2048)$. Do you remember that to get $C(M,N)$ we need $A(M,K)$ and $B(K,N)$? In other words, the second dimension of A and first dimension of B need to be equal. To achieve that, we need to transpose B. The formula becomes now: $C = A \times B^T$. The dimensions are ok now: $A(5,2048) \times B(2048, 512) = C(5, 512)$. Okay, so we would like to use cuBLAS now to compute the C.

But cuBLAS expects column-major format of A and B. Row-major transposed will give us column-major. So let's transpose the formula $C = A \times B^T$, using the property $C^T=B^T \times A^T$ and we get $C^T = (B^T)^T \times A^T$. From the third property of the matrix above, we know that a transposition of a transposition is equal to the original matrix, so we simplify $(B^T)^T$ to just $B$. The final formula is: $C^T = B \times A^T$. Let's check if dimensions are still correct. $B (512, 2048) \times A^T(2048, 5) = C^T(512, 5)$. The result dimensions seem inverse of what we wanted to get -- (5, 512) -- but notice that we still talk about $C^T$. The actual $C$, the output of the cublasGemmEx, is not transposed, so the final dimension is correct (5, 512). Now the code:

```cpp
cublasGemmEx(cublas_handle, CUBLAS_OP_T, CUBLAS_OP_N, KV_DIM, num_active_slots, EMBEDDING_LENGTH, &k_proj_alpha, weights.w_k[layer], CUDA_R_16BF, EMBEDDING_LENGTH, rms_norms, CUDA_R_16BF, EMBEDDING_LENGTH, &k_proj_beta, k_proj_batched_buffer, CUDA_R_16BF, KV_DIM, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
```

I want to preempt the last confusion you might have if you actually dig into the code. The flags `CUBLAS_OP_T` and `CUBLAS_OP_N` tell the cuBLAS which matrices to transpose. And we just derived the formula $C^T=B \times A^T$, so why do we now tell the cuBLAS to transpose the first matrix $B$? To understand it, think about column- / row-major again. From cuBLAS perspective, our row-major $B$ is transposed $B^T$, because cuBLAS reads it as if it were column-major. So we need to tell cuBLAS to transpose it, to get back the $B$ we derived. Similarly, since we derived that the second argument should be $A^T$, and cuBLAS reads row-major $A$ as a column-major $A^T$, then don't transpose it again, because it's how we wanted to provide it to the cublasGemmEx. Q.E.D. :D

> I will publish this section in slightly different form in [Paged Out! Issue #9 in the article "The cuBLAS transposition trick"](https://pagedout.institute/)


## Prefill vs decode

An interesting fact about LLM inference is that it's not exactly the same process for the first predicted token vs all the next tokens. To predict the first token, you need to process all the input tokens. The input tokens are user's prompt or the chat history. All the computation that we need to do to get the first predicted token is called _prefill_. Everything that will happen after is called _decode_. 

Most of the computation, like Q projection, attention, attention scores, feed-forward (MLP) is thrown away as soon as it's passed to the next operation - both in prefill and decode. The useful mental model is that the only thing that you preserve at every stage of LLM inference is K projection, V projection and what is the last generated token. That's it. There are interesting implications of it, for instance - you could stop the inference, copy your K and V projections and last generated token, restart the server, load them into the server and use the last generated token as the input and you'd get the same next token predicted, as in the original server instance. I hope some of you challenge my claim and actually test it - let me know if you do :D

## Why KV cache exists

We can reuse some parts of the computation results to predict the next tokens. You don't have to reuse the results, but they don't change so computing them again and again is a pure waste. You already know that the only data that gets moved forward in the computation is K and V projection and last generated token. If we generate 1 token at the same time, both K and V are vectors of bfloat16s and last generated token is a single int. If we generate more tokens at the same time - in other words, if we do batching - both K and V are matrices of bfloat16s and last generated tokens is a vector of ints.

When we process a token, regardless of whether it's prefill or decode, from perspective of data we preserve (K, V projections and last generated token) it looks the same:

0. ...
1. Compute K projection using last generated token
2. Store it
3. Compute V projection using last generated token
4. Store it
5. ...
6. Use all K projections and all V projections to compute attention
7. ...
8. Generate new token
9. Store it as last generated token

Let's say we don't store K and V projection for current token. It would mean that we need to compute all K and V projections for the current and all previous tokens before we can compute attention for current token. Again, pure waste. That's the reason why store the K and V projections. It's just a record of all previous K and V projections. You don't modify it during the LLM inference. You just append to it, with every processed token. The name of this K and V projections storage is KV cache.

## Attention

Attention is an important part of LLM inference. It's where you do a lot of matrix multiplication using Q, K and V projections you computed earlier. A basic formula for scaled dot-product attention that comes from a paper [Attention is all you need](https://arxiv.org/pdf/1706.03762) is:

$$\text{Attention}(Q,K,V)=\text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$$

< TODO: this section is difficult to write, but I want to make it good and useful to you. Putting code here now, and want to come back to writing it Later™ >


```cpp
for (int i = 0; i < NUM_Q_HEADS; ++i)
{
    int k_head_idx = i / GQA_Q_TO_K_RATIO; // i / 4 <- it means 4 Q heads uses the same 1 K head
    __nv_bfloat16 *q_head = q_proj + i * HEAD_DIM;
    __nv_bfloat16 *k_head = k_proj[layer] + k_head_idx * HEAD_DIM;
    __nv_bfloat16 *attn_score_head = attn_scores + input_tokens.size() * input_tokens.size() * i;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_T,
                                                    CUBLAS_OP_N,
                                                    input_tokens.size(),
                                                    input_tokens.size(),
                                                    HEAD_DIM,
                                                    &attn_alpha,
                                                    k_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    q_head,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    &attn_beta,
                                                    attn_score_head,
                                                    CUDA_R_16BF,
                                                    input_tokens.size(),
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
}

// causal mask, softmax and then attention scores * V
        // attn scores * V
        // (32, num_tok, num_tok) * (num_tok, 512)
        // GQA - 4 Q heads share 1 V head
        // attn_scores dim (32, num_tok, num_tok)
        // attn_scores head dim (num_tok, num_tok)
        // V dim (num_tok, 512)
        // NUM_V_HEADS is 8 -> 512 / 8 = 64
        // V_head dim (num_tok, 64)
        // output head dim: scores head * V head -> (num_tok, num_tok) * (num_tok, 64) = (num_tok, 64)
        // in total 32 output heads: so (num_tok, 64 * 32) = (num_tok, 2048)
for (int i = 0; i < NUM_Q_HEADS; ++i)
{
    int v_head_idx = i / GQA_ATTN_SCORES_TO_V_RATIO; // GQA, 4 Q heads for 1 V head
    // i * input_tokens.size() * input_tokens.size(),  because attn scores is (32, num_tok, num_tok)
    __nv_bfloat16 *attn_scores_head = attn_scores + i * input_tokens.size() * input_tokens.size();
    __nv_bfloat16 *v_head = v_proj[layer] + v_head_idx * HEAD_DIM;
    __nv_bfloat16 *output_attn_scores_head = attn_scores_v + i * HEAD_DIM;

    cublasStatus_t attn_score_status = cublasGemmEx(cublas_handle,
                                                    CUBLAS_OP_N,
                                                    CUBLAS_OP_N,
                                                    HEAD_DIM,
                                                    input_tokens.size(),
                                                    input_tokens.size(),
                                                    &attn_scores_v_alpha,
                                                    v_head,
                                                    CUDA_R_16BF,
                                                    KV_DIM,
                                                    attn_scores_head,
                                                    CUDA_R_16BF,
                                                    input_tokens.size(),
                                                    &attn_scores_v_beta,
                                                    output_attn_scores_head,
                                                    CUDA_R_16BF,
                                                    EMBEDDING_LENGTH,
                                                    CUBLAS_COMPUTE_32F,
                                                    CUBLAS_GEMM_DEFAULT);
}
```

## GQA

Unlike in a multi-head attention, one of the first attention methods, in [group-query attention (GQA)](https://arxiv.org/pdf/2305.13245) multiple query heads share the same key and value head. In our case, it's like: 4 query heads use the same 1 key and value head. See the code with computation K/V heads above.

< TODO describe in more details >

## SiLU

[SiLU](https://arxiv.org/pdf/1702.03118) is an activation function used in our reference LLM. It introduces "non-linearity" into a model. It means that weights can be zeroed when they are not needed. It helps in training models. Almost all machine learning models have them in their architecture, even the simplest multi-layer perceptrons. In fact, as far as I remember, models couldn't generalize good enough without activation functions. SiLU is similar to ReLU, but when negative values approach 0, there are not zeroed, but get small negative value instead. 

```cpp
__global__ void siluKernel(__nv_bfloat16 *a, __nv_bfloat16 *b)
{
    int workIndex = threadIdx.x + blockIdx.x * 8192;
    for (int i = 0; i < 8192; i += 1024)
    {
        a[workIndex + i] = (__nv_bfloat16)((float)a[workIndex + i] * (1 / (1 + expf(-(float)a[workIndex + i]))) * (float)b[workIndex + i]);
    }
}

// in-place, overwriting a
void silu(__nv_bfloat16 *a, __nv_bfloat16 *b, int num_tokens)
{
    siluKernel<<<num_tokens, 1024>>>(a, b);
}
```

## Softmax

[Softmax](https://en.wikipedia.org/wiki/Softmax_function) is a function that normalizes all elements in a vector. This one is a first, "sequential" version of a softmax. We will derive and implement the "online" version of it later. 

$$ \sigma(v)=\frac{e^{v_i}}{\sum_{j=1}{K}e^{v_j}} $$

The way you can implement it is very similar to RMSNorm we implemented earlier.

```cpp
__global__ void softmaxKernel(__nv_bfloat16 *input, int num_tokens)
{
    // softmaxxing per head
    // might waste a lot of memory by hardcoding the size here but can't use num_tokens directly
    __shared__ float row[1024]; // row[0] will contain max value after the loop
    __shared__ float max_val;
    // find max of the row to subtract it for numerical stability
    int workIndex = blockIdx.x * num_tokens + threadIdx.x;
    __nv_bfloat16 token = input[workIndex];
    row[threadIdx.x] = (float)token;
    __syncthreads();

    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = fmaxf(row[threadIdx.x], row[threadIdx.x + i]);
        }
        __syncthreads();
    }
    if (threadIdx.x == 0)
    {
        max_val = row[0]; // so I don't need to allocate another shared value for max_val
    }
    __syncthreads();

    // turn into exp
    row[threadIdx.x] = expf((float)token - max_val);
    __syncthreads();

    // now I can compute the numerical stable sum, similar pattern - tree reduction
    // reusing row memory
    for (int i = 1; i < num_tokens; i = i * 2)
    {
        if (threadIdx.x % (i * 2) == 0 && threadIdx.x + i < num_tokens)
        {
            row[threadIdx.x] = row[threadIdx.x] + row[threadIdx.x + i];
        }
        __syncthreads();
    }

    input[workIndex] = (__nv_bfloat16)(expf((float)token - max_val) / row[0]);
}

// input are masked attention scores (NUM_Q_HEADS, num_tok, num_tok)
void softmax(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Softmax kernel not launched";
        return;
    }

    softmaxKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

## Causal mask

Every token can attend only to previous tokens. See [this good and straighforward explanation](https://outcomeschool.com/blog/causal-masking-in-attention), so you can code it by yourself. It's really helpful to see the diagrams.

< TODO write more >

```cpp
__global__ void causalMaskKernel(__nv_bfloat16 *input, int num_tokens)
{
    if (threadIdx.x + blockIdx.x * blockDim.x >= num_tokens * num_tokens * NUM_Q_HEADS)
    {
        return;
    }

    int column = threadIdx.x;
    int row = blockIdx.x % num_tokens;
    if (column > row)
    {
        input[blockIdx.x * num_tokens + threadIdx.x] = -HUGE_VALF;
    }
}

void causalMask(__nv_bfloat16 *input, int num_tokens)
{
    if (num_tokens > 1024)
    {
        std::cout << "Can't launch more than 1024 threads on RTX 5090, Causal mask kernel not launched";
        return;
    }

    causalMaskKernel<<<num_tokens * NUM_Q_HEADS, num_tokens>>>(input, num_tokens);
#ifdef DEBUG
    cudaError error = cudaGetLastError();
    if (error != cudaError::cudaSuccess)
    {
        std::cout << "CUDA last error: " << cudaGetLastError() << std::endl;
    }
#endif
}
```

## Argmax

Pick the token that has a highest score. 

```cpp
cudaMemcpy(embed_proj_cpu.data(), embed_proj, sizeof(__nv_bfloat16) * VOCAB_SIZE, cudaMemcpyDeviceToHost);
max_token = (float)embed_proj_cpu[0];
max_token_idx = 0;
for (int token_idx = 0; token_idx < VOCAB_SIZE; ++token_idx)
{
    if ((float)embed_proj_cpu[token_idx] > max_token)
    {
        max_token = embed_proj_cpu[token_idx];
        max_token_idx = token_idx;
    }
}
std::cout << "Output token: " << (float)max_token << ", token index: " << std::to_string(max_token_idx) << std::endl;
```

## Feed forward network

A [feed-forward network](https://en.wikipedia.org/wiki/Feedforward_neural_network) / fully connected network / multi-layer perceptron. Three linear layers and SiLU.

```cpp
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              HIDDEN_DIM,       // m
              1,                // n
              EMBEDDING_LENGTH, // k
              &gate_alpha,
              weights.mlp_gate_proj[layer],
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              rms_norms,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              &gate_beta,
              gate,
              CUDA_R_16BF,
              HIDDEN_DIM,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);

// (1, 2048) * (2048, 8192) -> (1, 8192)
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              HIDDEN_DIM,       // m
              1,                // n
              EMBEDDING_LENGTH, // k
              &up_alpha,
              weights.mlp_up_proj[layer],
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              rms_norms,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              &up_beta,
              up,
              CUDA_R_16BF,
              HIDDEN_DIM,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);

silu(gate, up, 1);

down = buf_2048_2;
cublasGemmEx(cublas_handle,
              CUBLAS_OP_T,
              CUBLAS_OP_N,
              EMBEDDING_LENGTH, // m
              1,                // n
              HIDDEN_DIM,       // k
              &down_alpha,
              weights.mlp_down_proj[layer],
              CUDA_R_16BF,
              HIDDEN_DIM,
              gate,
              CUDA_R_16BF,
              HIDDEN_DIM,
              &down_beta,
              down,
              CUDA_R_16BF,
              EMBEDDING_LENGTH,
              CUBLAS_COMPUTE_32F,
              CUBLAS_GEMM_DEFAULT);
```

## Buffer reuse

Sometimes, a buffer you allocate has the same size as a buffer that will be used later in the code. And sometimes, when second buffer starts to be needed after when data in the first buffer is not needed anymore (it was already used in a computation and won't be used anywhere else), then we can use the first buffer to write data, which we would write to the second buffer. This way, we can allocate less memory. It means we reuse the same buffer between two different places. We can safely do it, once we confirm by lifetime analysis that lifetimes of these two buffers don't overlap. See how it works in for `buf_2048_1` and `buf_2028_2` in [`src/main.cpp`](src/main.cpp)

< TODO write more and explain lifetimes analysis >

## Static batching

Instead of processing one request at a time, we process N requests. The pros: bigger throughput (more user requests processed at the same time). The cons: higher latency (all prompts in the batch need to wait until the longest prompt finishes processing, before being returned to the user). What you need is to basically in every place where we assumed there is a single token processed at a time, process multiple ones. Same goes for prefill - for multiple prompts, etc.

< TODO write more >

## Continuous batching

Solves the problem of needing to wait for a longest prompt in batch to be processed. It solves it by having slots in a batch. You fill prompts into slots. Once generation in a given slot is finished, the result from it is returned to the user. A prompt that awaits in a queue gets selected for the freshly emptied slot. The prompt goes through the prefill (all other elements in batch wait until it's finished). Once prefill is done, batching continues with all the elements of batch, including the new one.

< TODO write more >

## Online softmax

< TODO write more and write down the math derivation >

[Online Softmax: Cohere Labs talk - Slides](https://docs.google.com/presentation/d/1f4Ys4es1l8bfUFN1bJuy12c3zHIPelcNGgVgbyb1zac/edit?slide=id.g742e3e7cd_1_16#slide=id.g742e3e7cd_1_16)

[From Boltzmann-Gibbs distribution to numerically stable online softmax on GPU with proofs](https://jedrzej.maczan.pl/boltzmann_gibbs_to_softmax_on_gpu.pdf)

## Paged Attention

Incoming!

Memory management idea from operating systems used in LLM inference

## Paged KV cache

Incoming!

## Paged Attention CUDA kernel

Incoming!

Jędrzej Maczan, 2026, Apache License 2.0
