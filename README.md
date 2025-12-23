<h1>
  Multi-Speaker Monaural Speech Separation: A 3- to 4- Speaker Scaling Study
</h1>

<p>
  This repository contains a comprehensive pipeline for separating,
  transcribing, and diarizing multi-speaker speech mixtures. It documents a
  research study focused on scaling time-domain end-to-end separation models
  from a three-speaker baseline to a more challenging four-speaker scenario.
</p>

<h2>1. Project Overview</h2>
<p>
  The "Cocktail Party Problem" involves isolating individual voices from a
  single-channel (monaural) recording of overlapping speakers.
</p>

<p>This project implements and evaluates two primary models:</p>
<ul>
  <li>
    <strong>Model A:</strong> A 3-speaker separation baseline focused on
    high-fidelity reconstruction.
  </li>
  <li>
    <strong>Model B:</strong> A 4-speaker scaled model integrated with an
    Automatic Speech Recognition (ASR) and Diarization pipeline.
  </li>
</ul>

<hr />

<h2>2. Dataset Information</h2>
<p>
  The project utilizes the <strong>LibriSpeech ASR corpus</strong> (derived from
  public domain audiobooks) to create synthetic speech mixtures.
</p>
<ul>
  <li>
    <strong>Source Data:</strong> train-clean-100, dev-clean, and test-clean
    subsets from OpenSLR.
  </li>
  <li>
    <strong>Mixture Generation:</strong>
    <ul>
      <li>
        <strong>Speaker Count:</strong> Randomly selecting 3 or 4 unique
        speakers.
      </li>
      <li>
        <strong>Sampling Rate:</strong> Resampled to 8,000 Hz for efficiency.
      </li>
      <li><strong>Duration:</strong> 2.0-second audio segments.</li>
    </ul>
  </li>
  <li>
    <strong>Voice Activity Detection (VAD):</strong> An energy-based VAD
    algorithm filters silence by estimating a noise floor at the 10th percentile
    and using a 4x threshold to identify active speech.
  </li>
</ul>

<hr />

<h2>3. Model Architectures</h2>
<p>
  The separation engine for both models is based on
  <strong>Conv-TasNet</strong> (Convolutional Time-domain Audio Separation
  Network), which avoids the limitations of STFT-based masking by working
  directly in the time domain.
</p>

<h3>Model A: 3-Speaker Baseline (best_model_sfar.ipynb)</h3>
<ul>
  <li><strong>Target:</strong> Optimized for 3 overlapping speakers.</li>
  <li>
    <strong>Structure:</strong> Linear 1D-convolutional encoder, a Temporal
    Convolutional Network (TCN) separator, and a transposed 1D-convolutional
    decoder.
  </li>
  <li>
    <strong>Purpose:</strong> Establish stable training parameters and
    high-quality separation metrics.
  </li>
</ul>

<h3>
  Model B: 4-Speaker Scaling &amp; Pipeline
  (Team13_final_postprocess_ASR_clean_v2.ipynb)
</h3>
<ul>
  <li><strong>Target:</strong> Scaled to handle 4 overlapping speakers.</li>
  <li>
    <strong>Scaling Strategy:</strong> Utilizes Transfer Learning, initializing
    weights from the 3-speaker baseline and fine-tuning.
  </li>
  <li>
    <strong>Integrated Pipeline:</strong>
    <ul>
      <li>
        <strong>ASR:</strong> faster-whisper for high-speed transcription.
      </li>
      <li><strong>Diarization:</strong> resemblyzer for speaker embeddings.</li>
    </ul>
  </li>
</ul>

<hr />

<h2>4. Training Process</h2>
<p>
  Both models share a rigorous training workflow designed for
  permutation-independent tasks.
</p>
<ul>
  <li>
    <strong>Loss Function:</strong> Scale-Invariant Signal-to-Distortion Ratio
    (SI-SDR).
  </li>
  <li>
    <strong>Permutation Invariant Training (PIT):</strong> Resolves the "label
    ambiguity" problem by calculating loss for all possible permutations of
    output vs. ground truth.
  </li>
</ul>

<ul>
  <li>
    <strong>Hyperparameters:</strong>
    <ul>
      <li>Learning Rate: 1e-4</li>
      <li>Batch Size: 4</li>
      <li>Epochs: 50</li>
    </ul>
  </li>
  <li>
    <strong>Hardware:</strong> NVIDIA A100 or L4 GPUs in a Google Colab
    environment.
  </li>
</ul>

<hr />

<h2>5. Post-Processing &amp; Evaluation (result_analysis.ipynb)</h2>
<p>
  Post-separation enhancement is used to improve the signal quality before ASR
  and Diarization tasks.
</p>

<ul>
  <li>
    <strong>Wiener Filtering:</strong> Applied to predicted masks to further
    reduce cross-talk and residual noise.
  </li>
  <li>
    <strong>Metric:</strong> Performance is measured by SI-SDR improvement
    between the raw mixture and the separated tracks.
  </li>
  <li>
    <strong>Key Findings:</strong> Model A provides robust 3-speaker separation.
    Model B shows a "performance cliff" in 4-speaker scenarios but achieves an
    average SI-SDR improvement of ~4.01 dB.
  </li>
</ul>

<hr />

<h2>6. Repository Structure</h2>
<ul>
  <li>
    <strong>Team13_final_postprocess_ASR_clean_v2.ipynb:</strong> The main
    4-speaker pipeline (Data → Separation → ASR → Diarization).
  </li>
  <li>
    <strong>best_model_sfar.ipynb:</strong> 3-speaker baseline and VAD
    prototyping.
  </li>
  <li>
    <strong>result_analysis.ipynb:</strong> Evaluation, Wiener filtering, and Δ
    SI-SDR visualization.
  </li>
  <li>
    <strong>Team13_FinalReport_ML.pdf:</strong> Technical report detailing
    findings and scaling analysis.
  </li>
</ul>

<hr />

<h2>7. Setup and Usage</h2>

<h3>Installation</h3>
<pre>
pip install torch torchaudio numpy pandas tqdm matplotlib faster-whisper resemblyzer scikit-learn soundfile torchcodec</pre
>

<h3>Execution</h3>
<ul>
  <li>
    <strong>For 3-Speaker Baseline:</strong> Run
    <code>best_model_sfar.ipynb</code> for prototyping.
  </li>
  <li>
    <strong>For 4-Speaker Pipeline:</strong> Run
    <code>Team13_final_postprocess_ASR_clean_v2.ipynb</code> for end-to-end
    transcription and diarized table generation.
  </li>
  <li>
    <strong>For Evaluation:</strong> Use <code>result_analysis.ipynb</code> to
    calculate SI-SDR improvements and apply post-filtering.
  </li>
</ul>
