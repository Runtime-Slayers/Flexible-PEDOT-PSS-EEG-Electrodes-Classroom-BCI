# BT54 — Flexible Dry EEG Electrodes via Conductive Hydrogel-PEDOT:PSS Nanocomposite: Classroom-Ready Brain Monitoring

**Domain:** Electronics / Biomedical Engineering / Materials Science / Neurotechnology  
**Date:** 2026-02-27  
**Status:** Brainstorming  
**Novelty Level:** ★★★★★ (Very High)  
**Feasibility:** ★★★☆☆ (Moderate — requires materials synthesis)

---

## PART A — WHAT & WHY

### A1. The Problem

Current EEG systems used in educational neuroscience research rely on either:
1. **Wet Ag/AgCl electrodes** — gold standard but require conductive gel, skin abrasion, 30+ min setup, trained technician, and dry out after 2-3 hours
2. **Commercial dry electrodes** — rigid metal pins that are uncomfortable, produce high impedance (>50 kΩ), and generate motion artifacts during normal classroom movement

Neither is suitable for longitudinal classroom studies where students need to wear EEG naturally for 6+ hours across multiple days without a technician present.

**The gap:** No electrode technology combines gel-level signal quality (< 5 kΩ impedance) with dry-electrode convenience (zero prep, self-applicable) in a flexible, comfortable form factor that maintains stable contact during natural classroom activities.

### A2. Why It Matters

| Stakeholder | Pain Point |
|---|---|
| Neuroscience researchers | Limited to 1-2 hour lab sessions; can't study real classrooms |
| Students (subjects) | Discomfort, social stigma of visible EEG caps |
| Educational neuroscience | Ecological validity gap between lab and real-world |
| Wearable EEG companies | Customer complaints about discomfort and signal quality |
| Clinical EEG (long-term monitoring) | Skin irritation from prolonged gel contact |

### A3. Research Gap

| Existing Work | Limitation |
|---|---|
| Ag/AgCl wet electrodes | Require gel, dry out in 2-3h, skin prep needed |
| Rigid dry electrodes (g.tec) | High impedance (50-200 kΩ), uncomfortable pins |
| PEDOT:PSS thin films | Brittle, poor mechanical durability |
| Carbon nanotube arrays | Expensive, inconsistent fabrication |
| Conductive textile electrodes | High impedance variability, poor high-frequency response |
| Self-adhesive gel patches | Single-use, expensive, skin irritation |

**Our innovation:** A flexible nanocomposite electrode using PEDOT:PSS embedded in a self-moisturizing PVA hydrogel matrix with silver nanowire reinforcement, achieving < 5 kΩ impedance without skin preparation, self-adhesive through hydrogel tack, and sustainable signal quality for 8+ hours.

### A4. Core Hypothesis

> *A PEDOT:PSS/PVA hydrogel nanocomposite electrode with AgNW reinforcement can achieve skin-electrode impedance < 5 kΩ at 10 Hz without skin preparation, maintain SNR > 20 dB for alpha-band EEG over 8 hours of continuous wear, and withstand > 10,000 flex cycles without impedance degradation > 10%.*

---

## PART B — TECHNICAL APPROACH

### B1. Mathematical Framework

#### Electrode-Skin Interface Model

**Equivalent circuit impedance:**

$$Z_{total}(f) = R_{lead} + \frac{R_{ct}}{1 + j2\pi f R_{ct} C_{dl}} + \frac{R_{gel}}{1 + j2\pi f R_{gel} C_{gel}} + R_{skin}$$

Where:
- $R_{lead}$: lead resistance (~1 Ω for AgNW)
- $R_{ct}$: charge transfer resistance at electrode-electrolyte interface
- $C_{dl}$: double-layer capacitance (~10 µF/cm² for PEDOT:PSS)
- $R_{gel}$: hydrogel bulk resistance
- $C_{gel}$: hydrogel dielectric capacitance

**PEDOT:PSS conductivity (percolation model):**

$$\sigma = \sigma_0 \left(\frac{\phi - \phi_c}{1 - \phi_c}\right)^t$$

Where $\phi$ is PEDOT:PSS volume fraction, $\phi_c \approx 0.05$ is percolation threshold, $t \approx 2.0$ is critical exponent.

**Hydrogel water retention (Flory-Rehner theory):**

$$\ln(1 - v_p) + v_p + \chi v_p^2 = -V_1 n_e (v_p^{1/3} - v_p/2)$$

Where $v_p$ is polymer volume fraction, $\chi$ is Flory-Huggins interaction parameter, $V_1$ is molar volume of water, $n_e$ is effective crosslink density.

#### Signal Quality Metrics

**Signal-to-Noise Ratio:**

$$SNR_{dB} = 10 \log_{10}\left(\frac{P_{signal}}{P_{noise}}\right) = 10 \log_{10}\left(\frac{\sigma^2_{EEG}}{V_{thermal}^2 + V_{1/f}^2 + V_{motion}^2}\right)$$

**Thermal noise (Johnson-Nyquist):**

$$V_{thermal} = \sqrt{4 k_B T |Z| \Delta f}$$

**Signal quality index over time:**

$$SQI(t) = \frac{SNR(t)}{SNR(t_0)} \cdot \frac{|Z(t_0)|}{|Z(t)|}$$

### B2. Material Architecture

```
Layer Structure (cross-section, ~500 µm total):

┌──────────────────────────────────────────────────┐
│  Protective Overlay (10 µm silicone)             │  ← Moisture barrier
├──────────────────────────────────────────────────┤
│  Flexible PCB trace (Cu/PI, 25 µm)              │  ← Signal routing
├──────────────────────────────────────────────────┤
│  Silver Nanowire mesh (5 µm)                     │  ← Current collection
├──────────────────────────────────────────────────┤
│  PEDOT:PSS / PVA Hydrogel Nanocomposite          │  ← Active electrode
│  (300 µm)                                        │     material
│  ┌─ PEDOT:PSS (15 wt%) ──────────────────┐      │
│  │  ┌─ AgNW (2 wt%) ─────────────┐       │      │
│  │  │  ┌─ PVA hydrogel matrix ─┐  │       │      │
│  │  │  │  (83 wt%, 60% water) │  │       │      │
│  │  │  └───────────────────────┘  │       │      │
│  │  └─────────────────────────────┘       │      │
│  └────────────────────────────────────────┘      │
├──────────────────────────────────────────────────┤
│  Self-adhesive hydrogel skin interface (100 µm)  │  ← Gentle tack
│  (PVA-borax reversible crosslinks)               │     (reusable)
└──────────────────────────────────────────────────┘
     ↕ SKIN CONTACT SURFACE
```

### B3. Python Implementation

```python
"""
BT54 - Flexible Dry EEG Electrode Simulator
PEDOT:PSS / PVA Hydrogel Nanocomposite
Electrode-skin impedance and signal quality modeling
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from scipy import optimize
import warnings
warnings.filterwarnings('ignore')


@dataclass
class MaterialProperties:
    """Nanocomposite material parameters."""
    # PEDOT:PSS
    pedot_conductivity_base: float = 1000.0  # S/m (pure PEDOT:PSS film)
    pedot_weight_fraction: float = 0.15
    pedot_percolation_threshold: float = 0.05
    pedot_critical_exponent: float = 2.0
    pedot_double_layer_cap: float = 10e-6  # F/cm²
    
    # Silver nanowires
    agnw_conductivity: float = 6.3e7  # S/m (bulk silver)
    agnw_weight_fraction: float = 0.02
    agnw_diameter_nm: float = 40
    agnw_length_um: float = 20
    
    # PVA Hydrogel
    pva_weight_fraction: float = 0.83
    water_content: float = 0.60  # 60% water in hydrogel
    pva_youngs_modulus: float = 50e3  # Pa (soft, skin-like)
    flory_chi: float = 0.49  # PVA-water interaction parameter
    crosslink_density: float = 1e-4  # mol/cm³
    
    # Composite
    thickness_um: float = 300
    area_cm2: float = 0.8  # ~10mm diameter electrode
    
    def effective_conductivity(self) -> float:
        """Calculate composite conductivity via percolation theory."""
        phi = self.pedot_weight_fraction + self.agnw_weight_fraction
        phi_c = self.pedot_percolation_threshold
        t = self.pedot_critical_exponent
        
        if phi <= phi_c:
            return 1e-6  # Below percolation — insulating
        
        sigma_0 = (self.pedot_conductivity_base * self.pedot_weight_fraction +
                    self.agnw_conductivity * self.agnw_weight_fraction * 0.01)  # Scaled
        
        sigma = sigma_0 * ((phi - phi_c) / (1 - phi_c)) ** t
        return sigma


class ElectrodeImpedanceModel:
    """Model electrode-skin impedance frequency response."""
    
    def __init__(self, material: MaterialProperties):
        self.mat = material
        
        # Circuit parameters
        self.R_lead = self._calc_lead_resistance()
        self.R_ct = self._calc_charge_transfer_resistance()
        self.C_dl = material.pedot_double_layer_cap * material.area_cm2
        self.R_gel = self._calc_gel_resistance()
        self.C_gel = self._calc_gel_capacitance()
        self.R_skin = 1000  # Ω (unprepared dry skin)
    
    def _calc_lead_resistance(self) -> float:
        """Lead resistance through AgNW mesh."""
        sigma = self.mat.agnw_conductivity * self.mat.agnw_weight_fraction
        L = self.mat.thickness_um * 1e-6  # m
        A = self.mat.area_cm2 * 1e-4  # m²
        return L / (sigma * A) if sigma > 0 else 1e6
    
    def _calc_charge_transfer_resistance(self) -> float:
        """Charge transfer at PEDOT:PSS / electrolyte(skin moisture) interface."""
        sigma = self.mat.effective_conductivity()
        # Inverse relationship with conductivity
        R_ct = 500 / (sigma / 100 + 1)  # Ω, empirical scaling
        return R_ct
    
    def _calc_gel_resistance(self) -> float:
        """Hydrogel bulk ionic resistance."""
        # Depends on water content and ionic species
        ionic_conductivity = 0.1 * self.mat.water_content  # S/m crude estimate
        L = self.mat.thickness_um * 1e-6
        A = self.mat.area_cm2 * 1e-4
        return L / (ionic_conductivity * A)
    
    def _calc_gel_capacitance(self) -> float:
        """Hydrogel dielectric capacitance."""
        epsilon_r = 80 * self.mat.water_content  # Water-dominated
        epsilon_0 = 8.854e-12
        A = self.mat.area_cm2 * 1e-4
        d = self.mat.thickness_um * 1e-6
        return epsilon_r * epsilon_0 * A / d
    
    def impedance(self, freq_hz: np.ndarray) -> np.ndarray:
        """Calculate total impedance magnitude vs frequency."""
        omega = 2 * np.pi * freq_hz
        
        # Electrode-electrolyte interface (R_ct || C_dl)
        Z_interface = self.R_ct / (1 + 1j * omega * self.R_ct * self.C_dl)
        
        # Hydrogel bulk (R_gel || C_gel)
        Z_gel = self.R_gel / (1 + 1j * omega * self.R_gel * self.C_gel)
        
        # Total series
        Z_total = self.R_lead + Z_interface + Z_gel + self.R_skin
        
        return np.abs(Z_total)
    
    def impedance_at_10hz(self) -> float:
        """Standard EEG impedance check at 10 Hz."""
        return float(self.impedance(np.array([10.0]))[0])
    
    def phase(self, freq_hz: np.ndarray) -> np.ndarray:
        """Phase angle vs frequency."""
        omega = 2 * np.pi * freq_hz
        Z_interface = self.R_ct / (1 + 1j * omega * self.R_ct * self.C_dl)
        Z_gel = self.R_gel / (1 + 1j * omega * self.R_gel * self.C_gel)
        Z_total = self.R_lead + Z_interface + Z_gel + self.R_skin
        return np.angle(Z_total, deg=True)


class SignalQualitySimulator:
    """Simulate EEG signal quality over extended wear time."""
    
    def __init__(self, impedance_model: ElectrodeImpedanceModel, 
                 material: MaterialProperties):
        self.Z_model = impedance_model
        self.mat = material
        self.kB = 1.38e-23  # Boltzmann constant
        self.T = 310  # Body temperature (K)
    
    def thermal_noise(self, impedance_ohm: float, 
                      bandwidth_hz: float = 50) -> float:
        """Johnson-Nyquist thermal noise voltage (V rms)."""
        return np.sqrt(4 * self.kB * self.T * impedance_ohm * bandwidth_hz)
    
    def motion_artifact(self, activity_level: float = 0.3) -> float:
        """Simulate motion artifact voltage based on activity level.
        activity_level: 0 = still, 1 = vigorous movement
        """
        # Electrode-skin relative motion generates triboelectric voltage
        base_artifact_uV = 5  # µV for still sitting
        motion_scale = 1 + 50 * activity_level ** 2
        return base_artifact_uV * motion_scale * 1e-6
    
    def flicker_noise(self, freq_hz: float = 10) -> float:
        """1/f noise at given frequency."""
        # Typical for PEDOT:PSS electrodes
        S_1f = 1e-14  # V²/Hz at 1 Hz
        return np.sqrt(S_1f / freq_hz)
    
    def water_loss_model(self, hours: float, 
                          humidity_pct: float = 50) -> float:
        """Model hydrogel water loss over time.
        Returns fraction of water remaining (0-1).
        """
        # Exponential decay with humidity-dependent rate
        k_evap = 0.03 * (1 - humidity_pct / 100)  # Rate constant (1/h)
        water_remaining = np.exp(-k_evap * hours)
        
        # Hydrogel also absorbs skin moisture (counteracts evaporation)
        skin_moisture_absorption = 0.02 * hours * (1 - water_remaining)
        
        return min(1.0, water_remaining + skin_moisture_absorption)
    
    def impedance_over_time(self, hours: np.ndarray, 
                             humidity_pct: float = 50) -> np.ndarray:
        """Track impedance drift over wear time."""
        impedances = []
        base_Z = self.Z_model.impedance_at_10hz()
        
        for h in hours:
            water_frac = self.water_loss_model(h, humidity_pct)
            
            # Impedance increases as water is lost
            Z_factor = 1 / (water_frac + 0.1)  # Prevent divide by zero
            Z_t = base_Z * Z_factor
            
            # First 30 min: impedance decreases as gel conforms
            if h < 0.5:
                settling_factor = 1.5 - h
                Z_t *= settling_factor
            
            impedances.append(Z_t)
        
        return np.array(impedances)
    
    def snr_over_time(self, hours: np.ndarray,
                       eeg_amplitude_uV: float = 20,
                       activity_level: float = 0.3,
                       humidity_pct: float = 50) -> np.ndarray:
        """Calculate EEG signal-to-noise ratio over wear duration."""
        Z_t = self.impedance_over_time(hours, humidity_pct)
        eeg_signal = eeg_amplitude_uV * 1e-6  # Convert to V
        
        snr_values = []
        for i, h in enumerate(hours):
            Z = Z_t[i]
            
            V_thermal = self.thermal_noise(Z)
            V_motion = self.motion_artifact(activity_level * (0.7 + 0.3 * np.random.random()))
            V_flicker = self.flicker_noise(10)
            
            noise_total = np.sqrt(V_thermal ** 2 + V_motion ** 2 + V_flicker ** 2)
            snr = 10 * np.log10((eeg_signal ** 2) / (noise_total ** 2))
            
            snr_values.append(snr)
        
        return np.array(snr_values)
    
    def signal_quality_index(self, hours: np.ndarray, **kwargs) -> np.ndarray:
        """Normalized signal quality index (0-1)."""
        snr = self.snr_over_time(hours, **kwargs)
        # Normalize: 40 dB = 1.0, 0 dB = 0.0
        sqi = np.clip(snr / 40, 0, 1)
        return sqi


class FlexDurabilityModel:
    """Model mechanical durability under repeated flexing."""
    
    def __init__(self, material: MaterialProperties):
        self.mat = material
        
        # Fatigue parameters
        self.fatigue_exponent = -0.08  # Power law fatigue
        self.initial_conductivity = material.effective_conductivity()
    
    def flex_test(self, n_cycles: int, flex_radius_mm: float = 5) -> Dict:
        """Simulate flex cycling test."""
        cycles = np.arange(0, n_cycles + 1, max(1, n_cycles // 100))
        
        # Conductivity degradation follows power law
        strain = self.mat.thickness_um * 1e-3 / (2 * flex_radius_mm)  # Engineering strain
        
        conductivity = []
        impedance_change = []
        
        for n in cycles:
            # Power law fatigue
            degradation = 1 - (strain * (n / 1000) ** 0.3) * 0.05
            degradation = max(0.5, degradation)  # Floor at 50% of original
            
            sigma_n = self.initial_conductivity * degradation
            conductivity.append(sigma_n)
            
            # Impedance increase (inverse of conductivity change)
            Z_change = (self.initial_conductivity / sigma_n - 1) * 100
            impedance_change.append(Z_change)
        
        # Find cycle count at 10% impedance increase
        Z_array = np.array(impedance_change)
        cycles_at_10pct = cycles[np.searchsorted(Z_array, 10)] if np.any(Z_array >= 10) else n_cycles
        
        return {
            'cycles': cycles,
            'conductivity': np.array(conductivity),
            'impedance_change_pct': Z_array,
            'cycles_at_10pct_degradation': cycles_at_10pct,
            'final_impedance_change_pct': Z_array[-1]
        }


class ElectrodeOptimizer:
    """Optimize nanocomposite formulation for target impedance."""
    
    def __init__(self, target_impedance_ohm: float = 5000):
        self.target_Z = target_impedance_ohm
    
    def objective(self, params: np.ndarray) -> float:
        """Optimization objective: minimize impedance deviation + material cost."""
        pedot_wt, agnw_wt, thickness_um, water_content = params
        
        # Constraints
        if pedot_wt < 0.05 or pedot_wt > 0.30:
            return 1e6
        if agnw_wt < 0.005 or agnw_wt > 0.05:
            return 1e6
        if thickness_um < 100 or thickness_um > 1000:
            return 1e6
        if water_content < 0.3 or water_content > 0.8:
            return 1e6
        if pedot_wt + agnw_wt > 0.40:
            return 1e6
        
        mat = MaterialProperties(
            pedot_weight_fraction=pedot_wt,
            agnw_weight_fraction=agnw_wt,
            pva_weight_fraction=1 - pedot_wt - agnw_wt,
            thickness_um=thickness_um,
            water_content=water_content
        )
        
        model = ElectrodeImpedanceModel(mat)
        Z_10 = model.impedance_at_10hz()
        
        # Multi-objective: impedance + durability proxy + cost
        impedance_error = ((Z_10 - self.target_Z) / self.target_Z) ** 2
        cost_penalty = agnw_wt * 50 + pedot_wt * 5  # AgNW is expensive
        thickness_penalty = (thickness_um / 500) ** 2  # Prefer thinner
        
        return impedance_error + 0.1 * cost_penalty + 0.05 * thickness_penalty
    
    def optimize(self, n_iterations: int = 200) -> Dict:
        """Grid search + local optimization."""
        best_result = None
        best_cost = float('inf')
        
        # Random search
        for _ in range(n_iterations):
            params = np.array([
                np.random.uniform(0.05, 0.25),  # PEDOT wt%
                np.random.uniform(0.005, 0.04),  # AgNW wt%
                np.random.uniform(150, 600),      # Thickness
                np.random.uniform(0.4, 0.75)      # Water content
            ])
            
            cost = self.objective(params)
            if cost < best_cost:
                best_cost = cost
                best_result = params.copy()
        
        # Local refinement
        try:
            result = optimize.minimize(self.objective, best_result,
                                       method='Nelder-Mead',
                                       options={'maxiter': 500})
            if result.fun < best_cost:
                best_result = result.x
                best_cost = result.fun
        except Exception:
            pass
        
        # Build optimal material
        opt_mat = MaterialProperties(
            pedot_weight_fraction=best_result[0],
            agnw_weight_fraction=best_result[1],
            pva_weight_fraction=1 - best_result[0] - best_result[1],
            thickness_um=best_result[2],
            water_content=best_result[3]
        )
        
        opt_model = ElectrodeImpedanceModel(opt_mat)
        
        return {
            'pedot_wt_pct': best_result[0] * 100,
            'agnw_wt_pct': best_result[1] * 100,
            'pva_wt_pct': (1 - best_result[0] - best_result[1]) * 100,
            'thickness_um': best_result[2],
            'water_content_pct': best_result[3] * 100,
            'impedance_10hz': opt_model.impedance_at_10hz(),
            'conductivity_S_m': opt_mat.effective_conductivity(),
            'optimization_cost': best_cost
        }


def run_full_simulation():
    """Execute complete flexible EEG electrode simulation."""
    print("=" * 70)
    print("BT54: FLEXIBLE DRY EEG ELECTRODES — PEDOT:PSS/PVA HYDROGEL")
    print("=" * 70)
    
    # Default material
    mat = MaterialProperties()
    model = ElectrodeImpedanceModel(mat)
    
    # Impedance spectrum
    print("\n╔══════════════════════════════════════════════════════════╗")
    print("║        ELECTRODE IMPEDANCE CHARACTERIZATION             ║")
    print("╠══════════════════════════════════════════════════════════╣")
    
    freqs = np.logspace(0, 3, 20)  # 1 Hz to 1 kHz
    Z = model.impedance(freqs)
    phase = model.phase(freqs)
    
    print(f"║ {'Freq (Hz)':>10} │ {'|Z| (kΩ)':>10} │ {'Phase (°)':>10}       ║")
    print(f"║{'─' * 10}─┼─{'─' * 10}─┼─{'─' * 10}───────║")
    for f_i, z_i, p_i in zip(freqs[::3], Z[::3], phase[::3]):
        print(f"║ {f_i:>10.1f} │ {z_i / 1000:>10.2f} │ {p_i:>10.1f}       ║")
    
    Z_10hz = model.impedance_at_10hz()
    print(f"╠══════════════════════════════════════════════════════════╣")
    print(f"║ Impedance @ 10 Hz:        {Z_10hz / 1000:>6.2f} kΩ                ║")
    print(f"║ Conductivity:             {mat.effective_conductivity():>6.1f} S/m                ║")
    print(f"║ Target:                    < 5.0 kΩ                    ║")
    print(f"║ Status: {'✓ PASS' if Z_10hz < 5000 else '✗ FAIL':>20s}                    ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Signal quality over time
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║     SIGNAL QUALITY OVER 8-HOUR WEAR (α-band)           ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    sqsim = SignalQualitySimulator(model, mat)
    hours = np.linspace(0, 8, 33)
    
    for activity, label in [(0.1, "Seated/Still"), (0.3, "Normal Class"), 
                             (0.6, "Active/Moving")]:
        snr = sqsim.snr_over_time(hours, activity_level=activity)
        sqi = sqsim.signal_quality_index(hours, activity_level=activity)
        
        print(f"║                                                          ║")
        print(f"║ Activity: {label:<15s}                                  ║")
        print(f"║ {'Hour':>6} │ {'SNR(dB)':>8} │ {'SQI':>5} │ {'Status':>10}         ║")
        
        for h_idx in range(0, len(hours), 8):
            h = hours[h_idx]
            s = snr[h_idx]
            q = sqi[h_idx]
            status = "Excellent" if q > 0.7 else "Good" if q > 0.5 else "Fair" if q > 0.3 else "Poor"
            print(f"║ {h:>6.1f} │ {s:>8.1f} │ {q:>5.2f} │ {status:>10s}         ║")
    
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Flex durability
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║            FLEX DURABILITY TEST                          ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    flex_model = FlexDurabilityModel(mat)
    flex_result = flex_model.flex_test(20000, flex_radius_mm=5)
    
    print(f"║ Flex radius:              5 mm                          ║")
    print(f"║ Total cycles tested:      20,000                        ║")
    print(f"║ Cycles at 10% degradation: {flex_result['cycles_at_10pct_degradation']:>6d}                    ║")
    print(f"║ Final impedance change:    {flex_result['final_impedance_change_pct'][-1]:>5.1f}%                     ║")
    
    checkpoints = [1000, 5000, 10000, 20000]
    for cp in checkpoints:
        idx = np.searchsorted(flex_result['cycles'], cp)
        if idx < len(flex_result['impedance_change_pct']):
            print(f"║ @ {cp:>6d} cycles: ΔZ = {flex_result['impedance_change_pct'][idx]:>5.1f}%"
                  f"{'':>24s}║")
    
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Optimization
    print(f"\n╔══════════════════════════════════════════════════════════╗")
    print(f"║         FORMULATION OPTIMIZATION                         ║")
    print(f"╠══════════════════════════════════════════════════════════╣")
    
    optimizer = ElectrodeOptimizer(target_impedance_ohm=3000)
    opt = optimizer.optimize(n_iterations=500)
    
    print(f"║ Optimal formulation:                                     ║")
    print(f"║   PEDOT:PSS:    {opt['pedot_wt_pct']:>6.1f} wt%                           ║")
    print(f"║   AgNW:         {opt['agnw_wt_pct']:>6.1f} wt%                           ║")
    print(f"║   PVA:          {opt['pva_wt_pct']:>6.1f} wt%                           ║")
    print(f"║   Thickness:    {opt['thickness_um']:>6.0f} µm                            ║")
    print(f"║   Water:        {opt['water_content_pct']:>6.1f} %                             ║")
    print(f"║ Result:                                                  ║")
    print(f"║   Z @ 10 Hz:    {opt['impedance_10hz'] / 1000:>6.2f} kΩ                          ║")
    print(f"║   σ:            {opt['conductivity_S_m']:>6.1f} S/m                           ║")
    print(f"╚══════════════════════════════════════════════════════════╝")
    
    # Comparison with existing electrodes
    print(f"\n--- Electrode Comparison ---\n")
    print(f"  {'Electrode Type':<30s} {'Z@10Hz (kΩ)':<15s} {'Prep Time':<12s} {'Wear (h)':<10s}")
    print(f"  {'-' * 67}")
    print(f"  {'Ag/AgCl (wet, gold std.)':<30s} {'1-3':<15s} {'30+ min':<12s} {'2-3':<10s}")
    print(f"  {'Dry metal pins (g.tec)':<30s} {'50-200':<15s} {'5 min':<12s} {'1-2':<10s}")
    print(f"  {'Conductive textile':<30s} {'20-100':<15s} {'2 min':<12s} {'4-6':<10s}")
    print(f"  {'BT54 PEDOT/PVA hydrogel':<30s} {f'{Z_10hz/1000:.1f}':<15s} {'0 min':<12s} {'8+':<10s}")
    
    return opt


if __name__ == '__main__':
    opt_result = run_full_simulation()
```

---

## PART C — EXPECTED RESULTS

### C1. Electrode Performance

| Metric | Wet Ag/AgCl | Dry Metal | **BT54 Hydrogel** |
|---|---|---|---|
| Impedance @ 10 Hz | 1-3 kΩ | 50-200 kΩ | **3-5 kΩ** |
| Prep time | 30+ min | 5 min | **0 min** |
| Wear duration | 2-3 h | 1-2 h | **8+ h** |
| SNR (alpha band) | 25-35 dB | 10-20 dB | **20-30 dB** |
| Motion artifact | Low | High | **Low-Medium** |
| Reusable | No | Yes | **Yes (50+ uses)** |
| Self-adhesive | No (tape needed) | No (headband) | **Yes** |

### C2. Durability

| Test | Target | Expected |
|---|---|---|
| Flex cycles (5mm radius) | > 10,000 | 15,000-20,000 |
| Impedance change @ 10K cycles | < 10% | 5-8% |
| Autoclave sterilization cycles | > 20 | 30+ |
| Shelf life (sealed) | > 6 months | 12+ months |
| Water retention (8h, 50% RH) | > 60% | 65-75% |

### C3. Cost Comparison

| Component | Cost per Electrode |
|---|---|
| PEDOT:PSS (Heraeus Clevios) | $0.15 |
| Silver nanowires | $0.40 |
| PVA + additives | $0.05 |
| Flexible PCB connector | $0.30 |
| **Total per electrode** | **$0.90** |
| Ag/AgCl (for comparison) | $0.50 (but single-use) |
| Over 50 uses: effective cost | **$0.018/use** |

---

## PART D — COMPARISON WITH EXISTING WORK

| Feature | Ag/AgCl Wet | g.tec Dry | Cognionics | EPOC Flex | **BT54 (Ours)** |
|---|---|---|---|---|---|
| Impedance | ★★★★★ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | **★★★★☆** |
| Comfort/Wearability | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★★☆ | **★★★★★** |
| Wear duration | ★★☆☆☆ | ★★☆☆☆ | ★★★☆☆ | ★★★☆☆ | **★★★★★** |
| Setup time | ★☆☆☆☆ | ★★★☆☆ | ★★★★☆ | ★★★★☆ | **★★★★★** |
| Self-adhesive | ✗ | ✗ | ✗ | ✗ | **✓** |
| Flexible | ✗ | ✗ | Partially | ✓ | **✓ (fully)** |
| Reusable | ✗ | ✓ | ✓ | ✓ | **✓ (50+)** |
| Cost/use | $0.50 | $5+ | $3+ | $2+ | **$0.018** |

---

## PART E — TOOLS & RESOURCES

### E1. Materials & Fabrication

| Material/Equipment | Supplier | Est. Cost |
|---|---|---|
| PEDOT:PSS (Clevios PH1000) | Heraeus | $80/100mL |
| PVA (MW 89-98K, 99%+ hydrolyzed) | Sigma-Aldrich | $30/500g |
| Silver nanowires (40nm × 20µm) | ACS Material | $120/100mL |
| Spin coater | Laurell WS-650 | Lab access |
| Screen printer (thick film) | DEK 248 | Lab access |
| Impedance analyzer | Gamry Reference 600+ | Lab access |
| Flexible PCB fabrication | PCBWay / OSH Park | $2/electrode |

### E2. Characterization Tools

| Tool | Purpose |
|---|---|
| EIS (Electrochemical Impedance Spectroscopy) | Impedance characterization |
| SEM/TEM | Nanostructure imaging |
| FTIR | Hydrogel composition verification |
| Tensile tester | Mechanical properties |
| EEG amplifier (OpenBCI) | Signal quality validation |

### E3. Publication Targets

| Venue | Type | Fit |
|---|---|---|
| Advanced Materials | Journal | ★★★★★ |
| Biosensors and Bioelectronics | Journal | ★★★★★ |
| IEEE Trans. Biomedical Engineering | Journal | ★★★★☆ |
| ACS Applied Materials & Interfaces | Journal | ★★★★☆ |
| Sensors and Actuators B | Journal | ★★★☆☆ |

### E4. Summary Metrics

| Dimension | Rating |
|---|---|
| Effort | 🔴 High (materials synthesis + characterization + EEG validation) |
| Difficulty | 🔴 High (nanocomposite formulation optimization) |
| Novelty | 🟢 Very High (self-moisturizing conductive hydrogel) |
| Impact | 🟢 Very High (enables real-world classroom neuroscience) |
| Time to Prototype | 4-6 months |
| Time to Publication | 8-12 months |
