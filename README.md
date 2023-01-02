# Core ML Stable Diffusion

Run Stable Diffusion on Apple Silicon with Core ML

<img src="assets/readme_reel.png">

This repository comprises:

- `python_coreml_stable_diffusion`, a Python package for converting PyTorch models to Core ML format and performing image generation with Hugging Face [diffusers](https://github.com/huggingface/diffusers) in Python
- `StableDiffusion`, a Swift package that developers can add to their Xcode projects as a dependency to deploy image generation capabilities in their apps. The Swift package relies on the Core ML model files generated by `python_coreml_stable_diffusion`

If you run into issues during installation or runtime, please refer to the [FAQ](#FAQ) section.


## <a name="example-results"></a> Example Results

There are numerous versions of Stable Diffusion available on the [Hugging Face Hub](https://huggingface.co/models?search=stable-diffusion). Here are example results from three of those models:

`--model-version` | [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) |  [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) |  [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) |
:------:|:------:|:------:|:------:
Output | ![](assets/a_high_quality_photo_of_an_astronaut_riding_a_horse_in_space/randomSeed_11_computeUnit_CPU_AND_GPU_modelVersion_stabilityai_stable-diffusion-2-base.png) | ![](assets/a_high_quality_photo_of_an_astronaut_riding_a_horse_in_space/randomSeed_13_computeUnit_CPU_AND_NE_modelVersion_CompVis_stable-diffusion-v1-4.png) | ![](assets/a_high_quality_photo_of_an_astronaut_riding_a_horse_in_space/randomSeed_93_computeUnit_CPU_AND_NE_modelVersion_runwayml_stable-diffusion-v1-5.png)
M1 iPad Pro 8GB Latency (s)     | 29 | 38 | 38 |
M1 MacBook Pro 16GB Latency (s) | 24 | 35 | 35 |
M2 MacBook Air 8GB Latency (s)  | 18 | 23 | 23 |

Please see [Important Notes on Performance Benchmarks](#important-notes-on-performance-benchmarks) section for details.


## <a name="converting-models-to-coreml"></a> Converting Models to Core ML

<details>
  <summary> Click to expand </summary>

**Step 1:** Create a Python environment and install dependencies:

```bash
conda create -n coreml_stable_diffusion python=3.8 -y
conda activate coreml_stable_diffusion
cd /path/to/cloned/ml-stable-diffusion/repository
pip install -e .
```

**Step 2:** Log in to or register for your [Hugging Face account](https://huggingface.co), generate a [User Access Token](https://huggingface.co/settings/tokens) and use this token to set up Hugging Face API access by running `huggingface-cli login` in a Terminal window.

**Step 3:** Navigate to the version of Stable Diffusion that you would like to use on [Hugging Face Hub](https://huggingface.co/models?search=stable-diffusion) and accept its Terms of Use. The default model version is [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). The model version may be changed by the user as described in the next step.

**Step 4:** Execute the following command from the Terminal to generate Core ML model files (`.mlpackage`)

```shell
python -m python_coreml_stable_diffusion.torch2coreml --convert-unet --convert-text-encoder --convert-vae-decoder --convert-safety-checker -o <output-mlpackages-directory>
```

**WARNING:** This command will download several GB worth of PyTorch checkpoints from Hugging Face.

This generally takes 15-20 minutes on an M1 MacBook Pro. Upon successful execution, the 4 neural network models that comprise Stable Diffusion will have been converted from PyTorch to Core ML (`.mlpackage`) and saved into the specified `<output-mlpackages-directory>`. Some additional notable arguments:

- `--model-version`: The model version defaults to [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4). Developers may specify other versions that are available on [Hugging Face Hub](https://huggingface.co/models?search=stable-diffusion), e.g. [stabilityai/stable-diffusion-2-base](https://huggingface.co/stabilityai/stable-diffusion-2-base) & [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5).


- `--bundle-resources-for-guernika`: Compiles all 4 models and bundles them along with necessary resources for text tokenization into `<output-mlpackages-directory>/Resources` which should provided as input to the Swift package. This flag is not necessary for the diffusers-based Python pipeline.

- `--chunk-unet`: Splits the Unet model in two approximately equal chunks (each with less than 1GB of weights) for mobile-friendly deployment. This is **required** for ANE deployment on iOS and iPadOS. This is not required for macOS. Swift CLI is able to consume both the chunked and regular versions of the Unet model but prioritizes the former. Note that chunked unet is not compatible with the Python pipeline because Python pipeline is intended for macOS only. Chunking is for on-device deployment with Swift only.

- `--attention-implementation`: Defaults to `SPLIT_EINSUM` which is the implementation described in [Deploying Transformers on the Apple Neural Engine](https://machinelearning.apple.com/research/neural-engine-transformers). `--attention-implementation ORIGINAL` will switch to an alternative that should be used for non-ANE deployment. Please refer to the [Performance Benchmark](#performance-benchmark) section for further guidance.

- `--check-output-correctness`: Compares original PyTorch model's outputs to final Core ML model's outputs. This flag increases RAM consumption significantly so it is recommended only for debugging purposes.

</details>

## <a name="image-generation-with-python"></a> Image Generation with Python

<details>
  <summary> Click to expand </summary>

Run text-to-image generation using the example Python pipeline based on [diffusers](https://github.com/huggingface/diffusers):

```shell
python -m python_coreml_stable_diffusion.pipeline --prompt "a photo of an astronaut riding a horse on mars" -i <output-mlpackages-directory> -o </path/to/output/image> --compute-unit ALL --seed 93
```
Please refer to the help menu for all available arguments: `python -m python_coreml_stable_diffusion.pipeline -h`. Some notable arguments:

- `-i`: Should point to the `-o` directory from Step 4 of [Converting Models to Core ML](#converting-models-to-coreml) section from above.
- `--model-version`: If you overrode the default model version while converting models to Core ML, you will need to specify the same model version here.
- `--compute-unit`: Note that the most performant compute unit for this particular implementation may differ across different hardware. `CPU_AND_GPU` or `CPU_AND_NE` may be faster than `ALL`. Please refer to the [Performance Benchmark](#performance-benchmark) section for further guidance.
- `--scheduler`: If you would like to experiment with different schedulers, you may specify it here. For available options, please see the help menu. You may also specify a custom number of inference steps by `--num-inference-steps` which defaults to 50.

</details>

## Image Generation with Swift

<details>
  <summary> Click to expand </summary>

### <a name="swift-requirements"></a> System Requirements
Building the Swift projects require:
- macOS 13 or newer
- Xcode 14.1 or newer with command line tools installed. Please check [developer.apple.com](https://developer.apple.com/download/all/?q=xcode) for the latest version.
- Core ML models and tokenization resources. Please see `--bundle-resources-for-guernika` from the [Converting Models to Core ML](#converting-models-to-coreml) section above

If deploying this model to:
- iPhone
  - iOS 16.2 or newer
  - iPhone 12 or newer
- iPad
  - iPadOS 16.2 or newer
  - M1 or newer
- Mac
  - macOS 13.1 or newer
  - M1 or newer

### Example CLI Usage
```shell
swift run StableDiffusionSample "a photo of an astronaut riding a horse on mars" --resource-path <output-mlpackages-directory>/Resources/ --seed 93 --output-path </path/to/output/image>
```
The output will be named based on the prompt and random seed:
e.g. `</path/to/output/image>/a_photo_of_an_astronaut_riding_a_horse_on_mars.93.final.png`

Please use the `--help` flag to learn about batched generation and more.

### Example Library Usage

```swift
import StableDiffusion
...
let pipeline = try StableDiffusionPipeline(resourcesAt: resourceURL)
pipeline.loadResources()
let image = try pipeline.generateImages(prompt: prompt, seed: seed).first
```
On iOS, the `reduceMemory` option should be set to `true` when constructing `StableDiffusionPipeline`

### Swift Package Details

This Swift package contains two products:

- `StableDiffusion` library
- `StableDiffusionSample` command-line tool

Both of these products require the Core ML models and tokenization resources to be supplied. When specifying resources via a directory path that directory must contain the following:

- `TextEncoder.mlmodelc` (text embedding model)
- `Unet.mlmodelc` or `UnetChunk1.mlmodelc` & `UnetChunk2.mlmodelc` (denoising autoencoder model)
- `VAEDecoder.mlmodelc` (image decoder model)
- `vocab.json` (tokenizer vocabulary file)
- `merges.text` (merges for byte pair encoding file)

Optionally, it may also include the safety checker model that some versions of Stable Diffusion include:

- `SafetyChecker.mlmodelc`

Note that the chunked version of Unet is checked for first. Only if it is not present will the full `Unet.mlmodelc` be loaded. Chunking is required for iOS and iPadOS and not necessary for macOS.

</details>

## <a name="performance-benchmark"></a> Performance Benchmark

<details>
  <summary> Click to expand </summary>

Standard [CompVis/stable-diffusion-v1-4](https://huggingface.co/CompVis/stable-diffusion-v1-4) Benchmark

|        Device                      | `--compute-unit`| `--attention-implementation` | Latency (seconds) |
| ---------------------------------- | --------------  | ---------------------------- | ----------------- |
| Mac Studio (M1 Ultra, 64-core GPU) | `CPU_AND_GPU`   |     `ORIGINAL`               |      9            |
| Mac Studio (M1 Ultra, 48-core GPU) | `CPU_AND_GPU`   |     `ORIGINAL`               |      13           |
| MacBook Pro (M1 Max, 32-core GPU)  | `CPU_AND_GPU`   |     `ORIGINAL`               |      18           |
| MacBook Pro (M1 Max, 24-core GPU)  | `CPU_AND_GPU`   |     `ORIGINAL`               |      20           |
| MacBook Pro (M1 Pro, 16-core GPU)  |    `ALL`        |     `SPLIT_EINSUM (default)` |      26           |
| MacBook Pro (M2)                   | `CPU_AND_NE`    |     `SPLIT_EINSUM (default)` |      23           |
| MacBook Pro (M1)                   | `CPU_AND_NE`    |     `SPLIT_EINSUM (default)` |      35           |
| iPad Pro (5th gen, M1)             | `CPU_AND_NE`    |     `SPLIT_EINSUM (default)` |      38           |


Please see [Important Notes on Performance Benchmarks](#important-notes-on-performance-benchmarks) section for details.

</details>

## <a name="important-notes-on-performance-benchmarks"></a> Important Notes on Performance Benchmarks

<details>
  <summary> Click to expand </summary>

- This benchmark was conducted by Apple using public beta versions of iOS 16.2, iPadOS 16.2 and macOS 13.1 in November 2022.
- The executed program is `python_coreml_stable_diffusion.pipeline` for macOS devices and a minimal Swift test app built on the `StableDiffusion` Swift package for iOS and iPadOS devices.
- The median value across 3 end-to-end executions is reported.
- Performance may materially differ across different versions of Stable Diffusion due to architecture changes in the model itself. Each reported number is specific to the model version mentioned in that context.
- The image generation procedure follows the standard configuration: 50 inference steps, 512x512 output image resolution, 77 text token sequence length, classifier-free guidance (batch size of 2 for unet).
- The actual prompt length does not impact performance because the Core ML model is converted with a static shape that computes the forward pass for all of the 77 elements (`tokenizer.model_max_length`) in the text token sequence regardless of the actual length of the input text.
- Pipelining across the 4 models is not optimized and these performance numbers are subject to variance under increased system load from other applications. Given these factors, we do not report sub-second variance in latency.
- Weights and activations are in float16 precision for both the GPU and the ANE.
- The Swift CLI program consumes a peak memory of approximately 2.6GB (without the safety checker), 2.1GB of which is model weights in float16 precision. We applied [8-bit weight quantization](https://coremltools.readme.io/docs/compressing-ml-program-weights#use-affine-quantization) to reduce peak memory consumption by approximately 1GB. However, we observed that it had an adverse effect on generated image quality and we rolled it back. We encourage developers to experiment with other advanced weight compression techniques such as [palettization](https://coremltools.readme.io/docs/compressing-ml-program-weights#use-a-lookup-table) and/or [pruning](https://coremltools.readme.io/docs/compressing-ml-program-weights#use-sparse-representation) which may yield better results.
- In the [benchmark table](performance-benchmark), we report the best performing `--compute-unit` and `--attention-implementation` values per device. The former does not modify the Core ML model and can be applied during runtime. The latter modifies the Core ML model. Note that the best performing compute unit is model version and hardware-specific.

</details>


## <a name="results-with-different-compute-units"></a> Results with Different Compute Units

<details>
  <summary> Click to expand </summary>

It is highly probable that there will be slight differences across generated images using different compute units.

The following images were generated on an M1 MacBook Pro and macOS 13.1 with the prompt *"a photo of an astronaut riding a horse on mars"* using the [runwayml/stable-diffusion-v1-5](https://huggingface.co/runwayml/stable-diffusion-v1-5) model version. The random seed was set to 93:

  CPU_AND_NE  |  CPU_AND_GPU  |  ALL  |
:------------:|:-------------:|:------:
![](assets/a_high_quality_photo_of_an_astronaut_riding_a_horse_in_space/randomSeed_93_computeUnit_CPU_AND_NE_modelVersion_runwayml_stable-diffusion-v1-5.png)  |  ![](assets/a_high_quality_photo_of_an_astronaut_riding_a_horse_in_space/randomSeed_93_computeUnit_CPU_AND_GPU_modelVersion_runwayml_stable-diffusion-v1-5.png) | ![](assets/a_high_quality_photo_of_an_astronaut_riding_a_horse_in_space/randomSeed_93_computeUnit_ALL_modelVersion_runwayml_stable-diffusion-v1-5.png) |

Differences may be less or more pronounced for different inputs. Please see the [FAQ](#faq) Q8 for a detailed explanation.

</details>

## FAQ

<details>
  <summary> Click to expand </summary>
<details>


<summary> <b> Q1: </b> <code> ERROR: Failed building wheel for tokenizers or error: can't find Rust compiler </code> </summary>

<b> A1: </b> Please review this [potential solution](https://github.com/huggingface/transformers/issues/2831#issuecomment-592724471).
</details>


<details>
<summary> <b> Q2: </b> <code> RuntimeError: {NSLocalizedDescription = "Error computing NN outputs." </code> </summary>

<b> A2: </b> There are many potential causes for this error. In this context, it is highly likely to be encountered when your system is under increased memory pressure from other applications. Reducing memory utilization of other applications is likely to help alleviate the issue.
</details>

<details>
<summary> <b> Q3: </b> My Mac has 8GB RAM and I am converting models to Core ML using the example command. The process is getting killed because of memory issues. How do I fix this issue? </summary>

<b> A3: </b>  In order to minimize the memory impact of the model conversion process, please execute the following command instead:

```bash
python -m python_coreml_stable_diffusion.torch2coreml --convert-vae-decoder -o <output-mlpackages-directory> && \
python -m python_coreml_stable_diffusion.torch2coreml --convert-unet -o <output-mlpackages-directory> && \
python -m python_coreml_stable_diffusion.torch2coreml --convert-text-encoder -o <output-mlpackages-directory> && \
python -m python_coreml_stable_diffusion.torch2coreml --convert-safety-checker -o <output-mlpackages-directory> &&
```

If you need `--chunk-unet`, you may do so in yet another independent command which will reuse the previously exported Unet model and simply chunk it in place:

```bash
python -m python_coreml_stable_diffusion.torch2coreml --convert-unet --chunk-unet -o <output-mlpackages-directory>
```

</details>

<details>
<summary> <b> Q4: </b> My Mac has 8GB RAM, should image generation work on my machine? </summary>

<b> A4: </b> Yes! Especially the `--compute-unit CPU_AND_NE` option should work under reasonable system load from other applications. Note that part of the [Example Results](#example-results) were generated using an M2 MacBook Air with 8GB RAM.
</details>

<details>
<summary> <b> Q5: </b> Every time I generate an image using the Python pipeline, loading all the Core ML models takes 2-3 minutes. Is this expected? </summary>

<b> A5: </b> Yes and using the Swift library reduces this to just a few seconds. The reason is that `coremltools` loads Core ML models (`.mlpackage`) and each model is compiled to be run on the requested compute unit during load time. Because of the size and number of operations of the unet model, it takes around 2-3 minutes to compile it for Neural Engine execution. Other models should take at most a few seconds. Note that `coremltools` does not cache the compiled model for later loads so each load takes equally long. In order to benefit from compilation caching, `StableDiffusion` Swift package by default relies on compiled Core ML models (`.mlmodelc`) which will be compiled down for the requested compute unit upon first load but then the cache will be reused on subsequent loads until it is purged due to lack of use.
</details>

<details>
<summary> <b> Q6: </b> I want to deploy <code>StableDiffusion</code>, the Swift package, in my mobile app. What should I be aware of?" </summary>

<b> A6: </b> [This section](#swift-requirements) describes the minimum SDK and OS versions as well as the device models supported by this package. In addition to these requirements, for best practice, we recommend testing the package on the device with the least amount of RAM available among your deployment targets. This is due to the fact that `StableDiffusion` consumes approximately 2.6GB of peak memory during runtime while using `.cpuAndNeuralEngine` (the Swift equivalent of `coremltools.ComputeUnit.CPU_AND_NE`). Other compute units may have a higher peak memory consumption so `.cpuAndNeuralEngine` is recommended for iOS and iPadOS deployment (Please refer to this [section](#swift-requirements) for minimum device model requirements). If your app crashes during image generation, please try adding the [Increased Memory Limit](https://developer.apple.com/documentation/bundleresources/entitlements/com_apple_developer_kernel_increased-memory-limit) capability to your Xcode project which should significantly increase your app's memory limit.
</details>

<details>
<summary> <b> Q7: </b> How do I generate images with different resolutions using the same Core ML models? </summary>

<b> A7: </b> The current version of `python_coreml_stable_diffusion` does not support single-model multi-resolution out of the box. However, developers may fork this project and leverage the [flexible shapes](https://coremltools.readme.io/docs/flexible-inputs) support from coremltools to extend the `torch2coreml` script by using `coremltools.EnumeratedShapes`. Note that, while the `text_encoder` is agnostic to the image resolution, the inputs and outputs of `vae_decoder` and `unet` models are dependent on the desired image resolution.
</details>

<details>
<summary> <b> Q8: </b> Are the Core ML and PyTorch generated images going to be identical? </summary>

<b> A8: </b> If desired, the generated images across PyTorch and Core ML can be made approximately identical. However, it is not guaranteed by default. There are several factors that might lead to different images across PyTorch and Core ML:


  <b> 1. Random Number Generator Behavior </b>

  The main source of potentially different results across PyTorch and Core ML is the Random Number Generator ([RNG](https://en.wikipedia.org/wiki/Random_number_generation)) behavior. PyTorch and Numpy have different sources of randomness. `python_coreml_stable_diffusion` generally relies on Numpy for RNG (e.g. latents initialization) and `StableDiffusion` Swift Library reproduces this RNG behavior. However, PyTorch-based pipelines such as Hugging Face `diffusers` relies on PyTorch's RNG behavior.

  <b> 2. PyTorch </b>

  *"Completely reproducible results are not guaranteed across PyTorch releases, individual commits, or different platforms. Furthermore, results may not be reproducible between CPU and GPU executions, even when using identical seeds."* ([source](https://pytorch.org/docs/stable/notes/randomness.html#reproducibility)).

  <b> 3. Model Function Drift During Conversion </b>

  The difference in outputs across corresponding PyTorch and Core ML models is a potential cause. The signal integrity is tested during the conversion process (enabled via `--check-output-correctness` argument to  `python_coreml_stable_diffusion.torch2coreml`) and it is verified to be above a minimum [PSNR](https://en.wikipedia.org/wiki/Peak_signal-to-noise_ratio) value as tested on random inputs. Note that this is simply a sanity check and does not guarantee this minimum PSNR across all possible inputs. Furthermore, the results are not guaranteed to be identical when executing the same Core ML models across different compute units. This is not expected to be a major source of difference as the sample visual results indicate in [this section](#results-with-different-compute-units).

  <b> 4. Weights and Activations Data Type </b>

  When quantizing models from float32 to lower-precision data types such as float16, the generated images are [known to vary slightly](https://lambdalabs.com/blog/inference-benchmark-stable-diffusion) in semantics even when using the same PyTorch model. Core ML models generated by coremltools have float16 weights and activations by default [unless explicitly overridden](https://github.com/apple/coremltools/blob/main/coremltools/converters/_converters_entry.py#L256). This is not expected to be a major source of difference.

</details>

<details>
<summary> <b> Q9: </b> The model files are very large, how do I avoid a large binary for my App? </summary>

<b> A9: </b> The recommended option is to prompt the user to download these assets upon first launch of the app. This keeps the app binary size independent of the Core ML models being deployed. Disclosing the size of the download to the user is extremely important as there could be data charges or storage impact that the user might not be comfortable with.

</details>

</details>
