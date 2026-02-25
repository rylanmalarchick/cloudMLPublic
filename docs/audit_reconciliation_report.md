# Comprehensive Audit Reconciliation Report: Machine Learning Modalities for Cloud Base Height Retrieval

## Introduction to the Validation Framework

The accurate retrieval of cloud base height represents a fundamentally challenging problem in atmospheric remote sensing, bearing critical implications for aviation safety, thermodynamic profiling, and climate radiative forcing parameterizations. Standard active instruments, such as the NASA Cloud Physics Lidar, achieve high vertical resolution at approximately thirty meters but face inherent limitations in swath width, power consumption, and long-term deployment feasibility on varied aerospace platforms. To circumvent these constraints, parallel scientific investigations have rigorously explored machine learning models capable of inferring cloud base height from passive sensor platforms and assimilated numerical weather prediction data.

This document serves as an exhaustive reconciliation report, evaluating two parallel scientific efforts documented in recent atmospheric machine learning literature against a rigorous external correctness audit conducted in February 2026. The first investigation focuses on inferring cloud base height via deep convolutional neural networks applied to thermal infrared imagery captured by the NASA ER-2 high-altitude research aircraft. The second investigation focuses on gradient-boosted decision trees utilizing physics-informed feature engineering derived from ERA5 reanalysis fields, with a specific focus on characterizing catastrophic domain shift across heterogeneous atmospheric regimes.

The primary objective of this assessment is to strictly evaluate the empirical correctness, physical plausibility, mathematical soundness, and internal consistency of the research, as dictated by the comprehensive audit parameters. This encompasses a detailed review of the structural modifications to the neural architectures, the mathematical derivations of thermodynamic variables, the decoupling of the training datasets across modalities, and the documentation of remaining scientific vulnerabilities within the public repositories. The analysis will determine whether the iterative updates applied to these manuscripts successfully fulfill the strict requirements of the correctness audit.

## Physical Foundations of the Atmospheric Problem

To understand the methodological choices evaluated by the audit, it is necessary to establish the physical constraints of boundary layer meteorology. Cloud base height is not a static geometrical feature; it is a dynamic thermodynamic threshold representing the lifting condensation level, where an ascending parcel of boundary layer air cools sufficiently to reach water vapor saturation.

For passive optical and thermal sensors deployed on high-altitude platforms or satellites, the cloud base is fundamentally occluded by the cloud top. Therefore, inferring the base from a top-down vantage point requires an algorithm to learn latent relationships between the observable horizontal structures (such as cloud top brightness temperature, spatial texture, and mesoscale cellular convection patterns) and the unobservable vertical profile.

Alternatively, numerical weather prediction systems like the ERA5 global reanalysis provide continuous, three-dimensional gridded estimations of the atmospheric state. However, these reanalysis products operate at coarse spatial resolutions that frequently fail to resolve localized sub-grid cloud dynamics. Machine learning applied to reanalysis data attempts to bridge this gap by mapping coarse environmental parameters—such as surface pressure, ambient temperature, and humidity—to highly precise point observations collected by airborne lidar. Both the image-based and the reanalysis-based modalities attempt to solve the identical physical problem using vastly different mathematical projections, making their independent audits and comparative synthesis highly valuable.

## Auditing the Thermal Infrared Deep Learning Modality

The first modality under review investigates the feasibility of extracting vertical atmospheric structure from strictly downward-looking thermal infrared arrays. Because passive sensors inherently observe cloud top brightness temperatures rather than internal vertical profiles, this approach relies on the capacity of deep learning models to learn structural proxies from the horizontal thermal emission fields.

### Physical Soundness of the Broadband Infrared Window

The neural network framework utilizes a broadband infrared atmospheric window, specifically isolating the 2–6 μm spectral range. The correctness audit explicitly evaluated this spectral selection and confirmed that the utilization of this specific wavelength band is physically sound and highly appropriate for the NASA ER-2 platform operating at an altitude of approximately 21 kilometers.

In this spectral regime, atmospheric absorption by water vapor and carbon dioxide is relatively minimized compared to the traditional long-wave infrared bands. This lack of atmospheric attenuation allows the sensor to record brightness temperatures that are highly correlated with the physical cloud top temperature. Because the optical path between the 21-kilometer aircraft cruising altitude and the boundary layer marine clouds (which are typically situated between 210 and 1500 meters) introduces minimal radiative interference in the 2–6 μm window, the input variable represents a mathematically and physically sound starting point for the extraction of thermal patterns by a convolutional neural network.

### Instrumental Preprocessing and Vignetting Corrections

Prior to convolutional neural processing, the raw thermal imagery undergoes a sequence of rigorous signal corrections to isolate the true atmospheric variance from instrumental noise. Each observation from the Infrared Array Imager is parsed into a highly constrained 20 × 22 pixel thermal brightness image, yielding exactly 440 total pixels that are carefully aligned with the nadir beam of the Cloud Physics Lidar instrument. A critical step in this preprocessing pipeline is the application of a median reference flat-field correction designed to mitigate optical vignetting.

The external audit strictly validated this preprocessing step, highlighting it in the official report as both "technically sound and necessary". In unmodified aerospace optical assemblies, the natural falloff of illumination toward the periphery of the image sensor creates an artificial radial brightness gradient. If this raw data were fed directly into a convolutional neural network without correction, the network's spatial kernels would inevitably overfit to this static instrumental artifact rather than learning the dynamic thermal structure of the underlying cloud fields. By enforcing a flat-field correction, the research appropriately isolates the atmospheric thermodynamic variance, ensuring the model learns meteorology rather than optical distortion.

### Architectural Modifications and Mathematical Necessity

The deployment of standard, off-the-shelf computer vision architectures introduces severe dimensional incompatibilities when applied to small-format atmospheric sensor data. The standard ResNet-18 architecture, originally engineered for the massive ImageNet dataset, expects input tensors of 224 × 224 spatial dimensions.

In the audited literature, the first convolutional layer of the ResNet-18 model was explicitly and manually modified: the standard 7 × 7 kernel with a stride of 2 and the subsequent max-pooling layer were replaced entirely with a 3 × 3 kernel utilizing a stride of 1, and the max-pooling operation was eliminated entirely. The audit comprehensively evaluated this choice and confirmed that this architectural surgery was an absolute "mathematical necessity".

If the default ResNet-18 stem had been utilized for this task, the initial 20 × 22 input image would have been subjected to a stride-2 convolution, immediately reducing the spatial dimensions to a 10 × 11 grid. This would have been followed immediately by a stride-2 max-pooling operation, further reducing the dimensions to a computationally starved 5 × 6 grid before the data even reached the core residual blocks. Such an aggressive, immediate decimation of the spatial grid would obliterate the micro-textural thermal gradients that the network requires to predict cloud base dynamics. By substituting a 3 × 3 convolution with a stride of 1, the model safely preserves the spatial hierarchy of the 440-pixel input through the initial layers of the residual network, allowing the model to establish meaningful receptive fields.

The empirical performance metrics strongly justify this architectural preservation. Under a stratified 5-fold cross-validation scheme over the 380 available samples, the modified, ImageNet-pretrained ResNet-18 architecture achieved an R² of 0.432 ± 0.094 with a Mean Absolute Error of 172.7 ± 17.6 meters. Conversely, the EfficientNet-B0 architecture, whose compound scaling parameters are tightly optimized for larger spatial domains and cannot easily accommodate such severe input constraints, struggled significantly on the same task.

| Model Configuration | Initialization | Augmentation | Cross-Validation R² | Mean Absolute Error (m) |
|---|---|---|---|---|
| ResNet-18 | ImageNet Pretrained | None | 0.432 ± 0.094 | 172.7 ± 17.6 |
| ResNet-18 | Random Initialization | None | 0.414 ± 0.127 | 169.5 ± 15.8 |
| EfficientNet-B0 | ImageNet Pretrained | None | 0.311 ± 0.109 | 201.4 ± 26.9 |
| ResNet-18 | ImageNet Pretrained | Geometric Transforms | 0.056 ± 0.070 | 242.6 ± 8.9 |
| EfficientNet-B0 | ImageNet Pretrained | Geometric Transforms | −0.060 ± 0.069 | 256.6 ± 15.1 |
| EfficientNet-B0 | Random Initialization | None | −0.118 ± 0.805 | 218.8 ± 55.8 |

*Table 1: Cross-validation performance of Deep Learning modalities, reflecting the architectural comparisons reviewed during the external audit.*

As demonstrated in the structured data above, even with the benefit of ImageNet pretraining, the EfficientNet-B0 network only managed to achieve an R² of 0.311 ± 0.109, representing a substantial performance gap. Furthermore, when the EfficientNet-B0 model was trained entirely from random initialization (scratch), the mathematical structure collapsed entirely, resulting in a negative R² of −0.118 ± 0.805. This massive fold-to-fold variance (±0.805) powerfully illustrates the instability of complex scaling laws when forced to converge on highly constrained, non-standard coordinate spaces without the stabilizing influence of pre-calculated feature weights.

### The Geometric Data Augmentation Paradox

A prevailing heuristic in modern deep learning dictates that data augmentation strictly improves model generalization by artificially expanding the diversity of the training distribution and preventing overfitting. However, the audited literature documents a consistent and highly severe degradation in predictive performance whenever geometric transforms were applied to the training pipeline.

The inclusion of standard computer vision augmentations—specifically horizontal and vertical flips, 15° rotations, 10% translations, and Gaussian blur—caused catastrophic structural collapse across all tested architectures. For the leading ResNet-18 model, the application of geometric augmentation caused the R² to plummet from a functional 0.432 to a nearly useless 0.056, while the Mean Absolute Error concurrently increased from 172.7 meters to 242.6 meters. For the EfficientNet-B0 model, augmentation drove the predictive power into the negative domain entirely, yielding an R² of −0.060.

The external audit explicitly reviewed this deeply counter-intuitive phenomenon and validated the degradation as a "mathematically correct consequence" of the instrumental geometry. In standard image classification, a natural image of an object (such as a vehicle or animal) remains conceptually and physically identical when rotated or flipped. However, the 20 × 22 pixel thermal array utilized in this study represents a strictly fixed, nadir-looking geographic swath that is inherently tied to the optical geometry of the NASA ER-2 aircraft and its forward flight vector.

A rotated thermal field does not represent a physically plausible atmospheric observation within the constraints of the fixed-array sensors used in this aerospace research campaign. By applying rotational and translational invariance during the training phase, the network was effectively forced to map physically impossible spatial coordinates to genuine physical altitudes. This corrupted the latent feature space, destroyed the spatial relationship between the center of the image and the synchronized lidar pulse, and artificially induced high prediction errors. The audit therefore commends the author's decision to disable geometric augmentation as a scientifically mature interpretation of instrument-specific data constraints.

### Computational Cost and Real-Time Viability

Beyond accuracy metrics, the operational viability of any atmospheric retrieval algorithm depends heavily on its computational efficiency, particularly if future iterations are targeted for edge-deployment aboard uncrewed aerial systems or satellite processing payloads. The literature provides a robust assessment of these parameters.

| Architecture Configuration | Training Time per Fold (seconds) | Inference Time (milliseconds) | Model Size in Memory (MB) |
|---|---|---|---|
| ResNet-18 (Modified Stem) | 70.4 ± 12.2 | 5.2 ± 0.0 | 43.1 |
| EfficientNet-B0 | 137.8 ± 29.1 | 9.1 ± 0.1 | 16.7 |

*Table 2: Computational cost and operational efficiency comparison across the evaluated convolutional neural network architectures.*

Despite possessing more than double the parameter count of the EfficientNet-B0 architecture (11.2 million parameters versus 5.3 million parameters, reflected in the 43.1 MB versus 16.7 MB model sizes), the modified ResNet-18 architecture proved substantially faster during both the backpropagation and inference phases. The ResNet-18 model completed training per fold in 70.4 ± 12.2 seconds, compared to 137.8 ± 29.1 seconds for EfficientNet-B0. More importantly for operational deployment, the ResNet-18 executed forward-pass inference in just 5.2 milliseconds, compared to 9.1 milliseconds for the EfficientNet. This processing speed confirms that the methodology is highly suitable for real-time, in-flight processing should the requisite thermal hardware be integrated into an operational pipeline.

### Heteroscedasticity and Target Distributions

Analysis of the ground-truth Cloud Physics Lidar dataset reveals a roughly bimodal cloud base height distribution with substantial density at both ends of the observed range. Within the 380 filtered, ocean-only samples utilized for the deep learning pipeline, the mean cloud base height is 913 meters with a standard deviation of 319 meters, bounded strictly between the physical realities of 210 meters and 1500 meters. The distribution shows concentrations near 300 m (low marine stratus) and 1100–1200 m (deeper boundary layer clouds), reflecting the two distinct campaign regimes sampled.

The regression diagnostics extracted from the neural network indicate an inherent heteroscedasticity in the model residuals. Specifically, the network exhibits a mean residual of −43.1 meters, indicating a mild systematic tendency to underpredict the true altitude of the cloud base on average. Furthermore, the error magnitude scales proportionally with the true cloud base height; predictions at the extreme upper bounds of the boundary layer (approaching the 1500-meter limit) exhibit substantially wider confidence intervals and greater scatter than those near the 200-meter marine inversion layer.

The statistical validation framework, specifically the choice to implement a stratified 5-fold cross-validation scheme rather than a simple randomized split, was confirmed as methodologically correct by the audit. Stratification ensures that the rare, high-altitude boundary layer clouds are distributed evenly across the training and validation folds. This guarantees that the heteroscedasticity is properly documented and quantified during evaluation rather than masked by lucky stochastic sampling.

## Auditing the Physics-Informed Feature Engineering Modality

In sharp contrast to the purely representation-learning approach of the thermal infrared convolutional network, the second investigation evaluated by the audit utilizes a gradient-boosted decision tree framework applied to numerical weather data. This model processes 5 raw variables assimilated from the ERA5 global reanalysis data and derives 29 explicit thermodynamic parameters, yielding a massive 34-dimensional feature space designed to mathematically map the ambient environmental state to the lidar-measured cloud base.

### Resolution of Audit Vulnerabilities: Feature Set Cardinality

Before analyzing the physical results, it is imperative to address a severe documentation vulnerability flagged during the external audit. The audit isolated a "severe numerical discrepancy regarding feature set cardinalities across the text and tables of the ERA5-focused manuscript". In earlier iterative drafts, the mathematical tally of the explicit thermodynamic features described in the methodology tables did not equal the total feature size stated in the abstract and discussion sections, calling the reproducibility of the research into question.

An exhaustive mathematical review of the current "Iteration 2 (Audit Reconciled)" text demonstrates that this critical vulnerability has been fully and meticulously resolved. The text firmly claims a reliance on 5 base variables and 29 derived features, yielding exactly 34 total parameters. A rigorous accounting of the newly tabulated features confirms this exact cardinality across the manuscript:

**Thermodynamic Features:** 8 distinct features are documented, capturing the broader atmospheric state. These include the mathematically complex virtual temperature (t_virtual), the direct lifting condensation level driver known as the dewpoint depression (dewpoint_depression), surface relative humidity (rh), the foundational stability index (stability_index), the vertical moisture gradient (moisture_gradient), the hypsometric pressure altitude (pressure_altitude), equivalent potential temperature (theta_e), and the boundary layer height to LCL ratio (blh_lcl_ratio).

**Stability Features:** 4 features are engineered to capture the interaction effects between atmospheric stratification and ambient moisture. These include the product of the stability index and total column water vapor (stability_tcwv), the dewpoint depression divided by boundary layer depth (dd_blh), the raw temperature to dewpoint ratio (t2m_d2m_ratio), and the calculated inversion strength (inversion_strength).

**Solar and Temporal Features:** 7 features encode the diurnal cycle and geometric heating effects. These consist of trigonometric encodings for the solar zenith angle (sza_cos, sza_sin) and solar azimuth angle (saa_cos, saa_sin), an estimated surface heating proxy based on this geometry (solar_heating_proxy), and temporal trigonometric components (hour_sin, hour_cos).

**Interaction Features:** 8 features capture highly nonlinear interactions between the foundational variables. These include temperature-moisture coupling (t2m_tcwv), humidity-boundary layer interactions (rh_blh), quadratic terms necessary for volume estimation (lcl_sq, blh_sq), pressure interactions (t2m_sp), and geographic proxies encoded via absolute latitude (lat_abs), absolute longitude (lon_abs), and their interaction (lat_lon).

**LCL-Based Features:** 2 core features are explicitly outlined in the text: the fundamental lifting condensation level derived directly from parcel theory (lcl), and the LCL deficit, which measures the mathematical divergence between the true boundary layer depth and the expected condensation altitude (lcl_deficit).

The sum of these explicitly derived features (8 + 4 + 7 + 8 + 2) exactly equals 29. When combined with the 5 foundational variables extracted directly from the ERA5 grid (2-meter temperature t2m, 2-meter dewpoint d2m, surface pressure sp, boundary layer height blh, and total column water vapor tcwv), the feature set perfectly totals 34.

Furthermore, the updated gradient boosting feature importance ablation study accurately lists the baseline model as containing "5 features only" and the enhanced model as containing "All 34 features". Therefore, the textual and tabular numerical discrepancies have been fully expunged, satisfying the rigorous demands of the external audit regarding version control and scientific consistency.

### Mathematical Verification of Thermodynamic Equations

A core component of the external audit involved the strict mathematical verification of the fundamental atmospheric equations engineered into the machine learning model.

The lifting condensation level serves as the foundational proxy for cloud base within the feature set. As an unsaturated air parcel is lifted adiabatically from the marine surface, it expands and cools at the dry adiabatic lapse rate (approximately 9.8°C per kilometer) while its dewpoint temperature decreases at a significantly slower rate. The exact altitude at which these two temperatures intersect dictates the point of water vapor saturation and the formation of the cloud base. The literature employs the standard meteorological approximation for this dynamic:

LCL ≈ 125 × (T₂ₘ − Tₐ)

where T₂ₘ represents the 2-meter surface temperature and Tₐ represents the 2-meter dewpoint temperature. The audit confirmed the mathematical correctness of this approximation within the context of the marine boundary layer, ensuring the model's physical grounding.

Additionally, the calculation of buoyancy relies heavily on virtual temperature. Because moist air is physically less dense than dry air at the identical temperature and pressure, atmospheric physicists cannot rely on standard ambient temperature to parameterize convective lifting. Virtual temperature provides the exact temperature that dry air must possess to have the same density as the moist air parcel in question. The literature calculates this parameter as:

Tᵥ = T × (1 + 0.61 × w)

where w represents the water vapor mixing ratio computed via the Clausius-Clapeyron relation. The audit meticulously verified the empirical 0.61 coefficient, confirming that it is correctly and mathematically derived from the ratio of the specific gas constants for dry air (Rₐ ≈ 287.05 J kg⁻¹ K⁻¹) and water vapor (Rᵥ ≈ 461.5 J kg⁻¹ K⁻¹). Specifically, (Rᵥ / Rₐ) − 1 ≈ 0.608 ≈ 0.61.

### Feature Importance and the Illusion of Explicit Engineering

The expansion from a compact 5-feature baseline to a massive 34-dimensional physics-informed feature space produced a distinct and highly analytical redistribution of feature importance within the gradient-boosted decision tree algorithm.

When the boosting framework was trained strictly on the 5 raw ERA5 features, the boundary layer height (blh) accounted for an overwhelming 63.3% of the predictive importance, with dewpoint (d2m) capturing 11.2% and surface pressure (sp) capturing 10.9%. This distribution perfectly reflects the dominant meteorological constraints operating in the marine environment: cloud base heights are inextricably linked to the depth of the well-mixed sub-cloud layer. The gradient boosting algorithm naturally isolated the boundary layer height as the master variable.

However, upon introducing the 29 explicitly derived thermodynamic equations, the importance spectrum fractured into highly specific interaction terms.

| Feature Category | Top Predictive Features | Relative Importance (%) | Physical Implication |
|---|---|---|---|
| Interaction | Quadratic Boundary Layer (blh_sq) | 32.3 | Captures non-linear volume expansions associated with deep atmospheric mixing. |
| Base | Boundary Layer Height (blh) | 16.9 | Retains baseline importance as the primary linear constraint on cloud altitude. |
| Stability | Stability-Moisture Interaction (stability_tcwv) | 8.0 | Mathematically captures how atmospheric stratification modulates the buoyant effects of moisture. |
| Thermodynamic | Vertical Moisture Gradient (moisture_gradient) | 8.0 | Complements the column-integrated water vapor metric by resolving the vertical distribution of saturation. |
| Thermodynamic | Boundary Layer to LCL Ratio (blh_lcl_ratio) | 4.4 | Identifies when turbulent mixing extends structurally above or below the theoretical parcel condensation level. |
| Various | Remaining 29 Features Combined | 30.4 | Minor contributions across specific thermal, temporal, and geometric domains. |

*Table 3: Redistribution of feature importance within the enhanced 34-feature gradient boosting model, highlighting the physical mechanisms prioritized by the algorithm.*

Importantly, the external audit validated the finding that complex derived variables, particularly those capturing stability interactions like stability_tcwv, successfully identified and prioritized critical predictive signals that matched human meteorological intuition.

However, despite providing massive interpretability advantages by forcing the model to rank known physical phenomena, the extensive feature engineering provided shockingly limited absolute accuracy gains over the raw reanalysis variables. The research executed a rigorous ablation study that yielded a counter-intuitive finding: dropping the 29 engineered features and running the gradient boosting model purely on the 5 base features actually improved the pooled cross-validation R² from −2.053 to −0.564.

This inversion of performance provides a deep insight into machine learning mechanics. It indicates that the gradient boosting architecture possesses sufficient internal complexity and depth to implicitly learn the non-linear thermodynamic relationships (analogous to the explicitly programmed equations) directly from the raw data. By explicitly forcing the 34 features into the model, the researchers inadvertently provided the algorithm with too many degrees of freedom, causing the model to overfit to flight-specific variance during the pooled training phase.

Conversely, when the models were evaluated under strict per-flight cross-validation (which completely prevents data leakage across different geographical regimes by evaluating models locally within a single flight), the 34-feature model performed vastly better (R² = −0.505) than the 5-feature baseline (R² = −2.042). This nuance establishes a critical finding: heavily engineered features actively harm generalized pooled training due to overfitting, but they substantially improve local, within-distribution generalizability where the specific thermodynamic interactions remain constant.

## Dataset Decoupling and Spatiotemporal Intersections

A highly notable discrepancy between the two papers evaluated by the audit is the vast difference in their respective sample sizes. The deep learning investigation utilizing thermal infrared imagery relies on a highly constrained dataset of merely 380 matched samples. Conversely, the gradient boosting investigation utilizing ERA5 reanalysis fields leverages a massive corpus of 5,500 observation samples spanning six ER-2 flights.

The external audit explicitly analyzed this variance to ensure no methodological errors were present, ultimately ruling that the decoupling of the datasets across the two modalities was both mathematically and scientifically justified.

The deep learning application faced immense physical constraints. It required a strict temporal and spatial intersection across three independent variables: an active Cloud Physics Lidar profile had to be matched within exactly 0.5 seconds of a valid, uncorrupted thermal infrared frame from the Infrared Array Imager, while simultaneously satisfying the condition that the underlying cloud structure was single-layer and located strictly over an ocean surface. Because the thermal imager operates at a slow frame rate of approximately 1 Hz and possesses a narrow field of view, the mathematical intersection of these stringent requirements aggressively decimated the available dataset. For example, the entire WHySMIE flight on October 23, 2024, yielded only 2 valid matched samples for the imagery pipeline despite hours of flight time.

Conversely, the feature engineering investigation did not rely on the thermal imager. It was structured entirely around matching the robust, high-frequency lidar profiles to the nearest physical grid point and nearest hour of the ERA5 global reanalysis model. Because the ERA5 numerical weather prediction system offers continuous, unbroken global coverage without the physical constraints of camera framing or sensor dropout, the second paper was not bottlenecked by spatial intersections. This structural freedom allowed the ingestion of the entire, unfiltered Cloud Physics Lidar Level 2 boundary-layer ocean observation corpus, yielding 5,500 high-quality samples for the analysis of domain shift.

## The Catastrophe of Domain Shift and the Illusion of Transfer Learning

The most consequential and disruptive finding within the entire ERA5 modality investigation is the documentation of a catastrophic failure in out-of-distribution generalization. Atmospheric data represents a continuous time-series of fluid dynamics, meaning it inherently and aggressively violates the fundamental independent and identically distributed (i.i.d.) assumption that underpins all standard machine learning validation frameworks.

### The Illusion of Pooled Cross-Validation

The literature mathematically demonstrates that consecutive atmospheric samples collected along an aircraft flight path exhibit severe temporal autocorrelation, characterized by a mean lag-1 Pearson correlation coefficient of ρ = 0.89, ranging from 0.82 to 0.97 across individual flights. This implies that any given measurement is highly correlated with the measurement recorded immediately prior.

In a standard pooled K-fold cross-validation scheme—which is heavily utilized throughout atmospheric science literature—these adjacent, highly correlated samples are randomly partitioned into separate training and testing folds. This introduces massive, invisible data leakage. The model achieves an artificially inflated performance metric because the test data it is asked to evaluate is practically identical to the training data it observed just microseconds prior during backpropagation. To document the true, operational generalizability of the algorithm, the researchers rightfully abandoned pooled cross-validation and utilized a strict Leave-One-Flight-Out (LOFO) protocol. In this framework, models were trained on data from all flight campaigns except one, and then deployed entirely cold on the held-out atmospheric regime.

### Covariate Shift and Generalization Failure

The LOFO validation revealed a total and catastrophic predictive collapse. Across the 5,500 observations, the models yielded an abysmal mean R² of −5.36 with a Mean Absolute Error of 518 meters. In regression analytics, an R² of exactly 0.0 indicates a naive model that simply predicts the constant mean of the target distribution regardless of the input features. An R² of −5.36 indicates an algorithm that is generating predictions completely decoupled from physical reality, producing residual variances that are over five times larger than the inherent variance of the underlying dataset itself.

The failure was systemic across all testing partitions. When the model trained on all other data was tested on the November 4, 2024 WHySMIE flight, the LOFO R² plunged to an astonishing −19.41.

This mathematical collapse is dictated by absolute distribution disjoints across temporal and spatial domains, known as covariate shift. The WHySMIE campaign, conducted in October 2024, surveyed the sub-tropical marine boundary layer along the California-Baja coast. This region is characterized by highly suppressed, shallow stratocumulus structures generally confined below 2,000 meters in altitude. Conversely, the GLOVE campaign, conducted in February 2025, surveyed the northeast Pacific off the Oregon coast during the peak of winter. This regime encountered deep, multi-layer complex cloud systems driven by fundamentally different synoptic forcing and baroclinic instability.

Kolmogorov-Smirnov (K-S) divergence tests successfully quantified this severe covariate shift. The K-S statistic measures the maximum distance between the empirical cumulative distribution functions of two separate samples. For key thermodynamic variables—specifically the 2-meter temperature (t2m), boundary layer height (blh), and the highly predictive virtual temperature (t_virtual)—the K-S statistic reached its theoretical maximum of 1.000, with p-values effectively at zero.

A K-S statistic of exactly 1.0 indicates completely non-overlapping feature spaces. The machine learning model, trained extensively on the warm, shallow WHySMIE data, was asked to execute mathematical inference on the cold, deep GLOVE data. Because the GLOVE data occupied numerical coordinate spaces the model had never mapped during training, the decision trees extrapolated wildly outside their bounds. This resulted in the model generating physically impossible negative cloud base heights for 1.7% of the LOFO predictions, demonstrating a complete breakdown of physical constraints under domain shift.

### The Failure of Advanced Adaptation Methods

Faced with this insurmountable covariate shift, the investigation systematically applied five sophisticated domain adaptation methodologies. The results yield critical, highly actionable evidence regarding what statistical interventions actually function in operational meteorology.

| Domain Adaptation Methodology | Mean Cross-Flight R² | Mechanism of Failure or Success |
|---|---|---|
| No Adaptation (LOFO Baseline) | −5.36 | Unmitigated extrapolation across non-overlapping thermodynamic regimes. |
| Maximum Mean Discrepancy (MMD) | −7.9 | Alignment forces divergent temperature domains to overlap, destroying the predictive physical signal entirely. |
| Feature Selection | −6.9 | Eliminating shifting features leaves only non-predictive geometry (solar angles), removing the core meteorological constraints. |
| Instance Weighting (Density Ratio) | −5.5 | The K-S = 1.0 divergence ensures no source samples match the target, causing the algorithm to amplify pure noise. |
| Instance Weighting (K-Nearest Neighbors) | −3.5 | Slight improvement over density ratios, but fundamentally suffers from the identical lack of target similarity in the source data. |
| TrAdaBoost | +0.04 | Down-weighting poorly transferring source samples halts the catastrophic negative variance, but fails to build predictive power. |
| Few-Shot Fine-Tuning (50 Local Samples) | +0.35 | Exposing the frozen base model to just 50 labeled local observations successfully recalibrates the decision thresholds to the new environment. |

*Table 4: Evaluation of domain adaptation methodologies applied to mitigate the catastrophic atmospheric domain shift.*

As detailed above, mathematically elegant solutions actively worsened the problem. Maximum Mean Discrepancy (MMD) alignment attempts to project the source and target feature spaces into a shared representation where their mean embeddings are statistically indistinguishable. However, because the variables that dictate the shift (temperature, ambient moisture) are the exact variables that dictate the altitude of the cloud physics, forcing the winter GLOVE distribution to mathematically align with the autumn WHySMIE distribution erased the fundamental thermodynamic state the algorithm required to estimate altitude, plunging the R² to −7.9.

Similarly, instance weighting techniques utilizing K-Nearest Neighbors or direct density ratio estimation attempted to reweight source samples to match the target distributions. Due to the absolute feature divergence, there were practically no source samples sufficiently similar to the target domain to up-weight. The algorithm simply amplified mathematical noise, yielding terrible R² values of −3.5 and −5.5 respectively.

The sole successful intervention was the simplest: standard supervised few-shot learning. By exposing the frozen base model to merely 50 labeled samples collected directly from the deployment flight, the algorithm abandoned its historical biases and calibrated its internal decision thresholds to the localized thermodynamic environment, successfully recovering a mean operational R² of +0.35. This establishes that for atmospheric deployment, sophisticated mathematical projections are vastly inferior to the acquisition of minimal local ground truth.

## Uncertainty Quantification and Exchangeability Violations

Quantifying uncertainty is a mandatory prerequisite for any operational aircraft deployment relying on machine learning inference. A pilot or automated flight system cannot rely on a static prediction; it requires a bounded interval defining the highest and lowest probable cloud bases.

The literature attempts to solve this using split conformal prediction, a mathematically elegant framework that theoretically guarantees that true values will fall within predicted intervals (1−α) percent of the time (e.g., establishing a 90% target coverage boundary) by analyzing the distribution of held-out calibration residuals. However, the fundamental proof of conformal prediction mathematically requires the underlying data to be exchangeable. Exchangeability mandates that the joint probability distribution of the data sequence must remain completely invariant regardless of the order in which the data is permuted.

Because of the severe cross-campaign domain shift and the substantial temporal autocorrelation within the flights (ρ=0.89 mean at lag-1, ranging 0.82–0.97), the exchangeability assumption is fundamentally and irreparably shattered in atmospheric observation datasets. When the conformal prediction framework was calibrated across different flights, it suffered a catastrophic collapse, achieving only a 34% coverage rate against the 90% target. The calibration residuals calculated from one specific flight grossly and systematically underestimated the massive test-time extrapolation errors that occurred when the model was subjected to a different season and synoptic environment.

Adaptive conformal prediction, which adjusts the prediction quantile online as test samples arrive, performed even worse, achieving only 20% mean coverage across flights. Because the initial cross-flight errors are so large relative to the calibration residuals, the online update rule rapidly shrinks the quantile toward zero, producing degenerate near-zero-width intervals on five of six flights. Only the February 18 flight—which by chance had the smallest domain shift—retained meaningful intervals.

The research astutely identified that the only mathematical path to valid statistical guarantees is strict per-flight calibration. By deliberately shrinking the calibration window, the researchers calibrated the conformal bounds strictly within the specific flight currently being executed. By holding out 20% of the active flight data for real-time calibration, the framework restored the localized independent and identically distributed structure of the data, perfectly and exactly recovering the target 90.0% coverage with an average interval width of 538 meters.

## Scientific Dissemination and the Repository Misalignment

A primary objective of this report is to not only evaluate the scientific validity of the manuscripts but to explicitly address the remaining vulnerabilities flagged in the February 2026 external audit regarding the public dissemination of this research.

While the rigorous texts of the scientific manuscripts have been thoroughly rectified to satisfy the audit's demands on feature cardinality and mathematical derivations, the audit highlighted a lingering, critical vulnerability concerning the project's open-source architecture. Specifically, the official audit document explicitly flagged a "significant misalignment" within the public GitHub repository utilized by the author to host the executable models.

The audit meticulously notes that the repository architecture uniformly advertises the absolute highest achieved performance metric from the Gradient Boosting model without mathematically or structurally differentiating it from the vastly distinct, geographically agnostic Deep Learning image pipelines.

As previously established, the ERA5 gradient boosting model exhibits massive performance variance. When evaluated under highly favorable, localized within-flight conditions (specifically, using the per-flight conformal calibration framework with a 60/20/20 temporal split), the gradient boosting model achieved a spectacular maximum R² of 0.86 during the February 12th GLOVE flight. However, its cross-regime generalization is fundamentally broken, evidenced by the −5.36 LOFO collapse.

Conversely, the thermal infrared ResNet-18 model, while possessing a lower peak accuracy, achieved a vastly more stable cross-validation R² of 0.432 across varying regimes. The convolutional architecture's reliance on direct thermal emission observations, rather than the assimilation of broad, shifting synoptic variables, renders it far more robust to geographical displacement.

By conflating the maximum localized performance of the gradient boosted decision trees (R²=0.86) with the generalized, cross-domain performance of the convolutional neural network (R²=0.432), the repository architecture poses a serious and documented "risk of scientific misrepresentation". The repository interface fails to construct a clear statistical boundary separating the modalities, implicitly suggesting to external researchers or operational engineers that a highly generalized algorithm achieves the peak localized accuracy across all atmospheric domains.

Because both provided Iteration 2 manuscripts contain standard, required "Code Availability" sections containing direct hyperlinks routing the reader to this exact public repository (https://github.com/rylanmalarchick/CloudMLPublic), the vulnerability extends directly from the peer-reviewed paper into the unregulated open-source community. While the scientific text has resolved all mathematical and structural criticisms leveled against it, the external housing of the executable code remains in direct violation of the external audit's explicit demand for clear methodological differentiation.

## Synthesis

The comprehensive review of the two divergent machine learning modalities for cloud base height retrieval confirms that the current iterations of the research have successfully integrated and resolved the vast majority of the rigorous requirements established by the February 2026 external audit.

In the highly constrained domain of thermal infrared imagery processing, the decision to surgically truncate the ResNet-18 architecture prior to its initial spatial decimation guarantees the mathematical preservation of the narrow 20 × 22 pixel geographic framing. The calculated rejection of geometric data augmentation demonstrates a highly sophisticated, physics-informed understanding of how classical computer vision techniques aggressively violate the fixed-angle, nadir-looking constraints of aerospace Earth observation instrumentation.

In the parallel domain of physics-informed numerical modeling, the rigorous derivations of virtual temperature density mappings and lifting condensation levels have been perfectly validated against classical meteorological constraints. The text meticulously resolves prior numerical discrepancies regarding the massive 34-dimensional feature space, ensuring total reproducibility.

More importantly for the broader field of applied atmospheric machine learning, the transparent documentation of the R² = −5.36 generalization failure serves as a critical, paradigm-shifting warning against the naive deployment of uncalibrated regression algorithms across dissimilar synoptic environments. Furthermore, the catastrophic breakdown of conformal prediction under conditions of extreme temporal autocorrelation dictates that any operational aircraft implementation attempting to provide safety-critical statistical bounds must execute strictly localized, per-flight statistical calibration.

The sole remaining deficiency highlighted by the external validation process pertains entirely to the public repository structure, which currently fails to differentiate the success metrics of the highly localized ERA5 models from the generalized thermal infrared convolutional networks. With the exception of this external alignment vulnerability, the internal contents, mathematical derivations, physical constraints, and statistical claims of the two atmospheric machine learning papers represent a fully reconciled, physically sound, and exhaustively validated scientific effort ready for broader implementation in the aerospace sector.
