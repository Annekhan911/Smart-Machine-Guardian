
import streamlit as st
import numpy as np
import pandas as pd
import scipy.io
import scipy.signal as sig
from scipy.stats import kurtosis, skew
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os

# ---------------- CONFIGURATION ----------------

st.set_page_config(
    page_title="Smart Machine Guardian",
    page_icon="🛡️",
    layout="wide"
)

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "model",
    "smart_machine_guardian_models.pkl"
)

# ---------------- LOAD MODELS ----------------

@st.cache_resource
def load_models():
    return joblib.load(MODEL_PATH)

try:
    bundle = load_models()
    severity_model = bundle["severity_model"]
    fault_model = bundle["fault_model"]
    FEATURES = bundle["features"]
    FS = bundle["sampling_rate"]
    SEGMENT_SIZE = bundle["segment_size"]
    SOS = bundle["filter_sos"]
except Exception as e:
    st.error(f"Could not load model bundle: {e}")
    st.stop()


# ---------------- FEATURE EXTRACTION ----------------

def extract_features(x, fs):
    x = np.asarray(x, dtype=float)
    x = x - np.mean(x)

    rms = np.sqrt(np.mean(x**2))
    peak = np.max(np.abs(x))
    crest = peak / (rms + 1e-12)

    spectrum = np.abs(np.fft.rfft(x))
    frequencies = np.fft.rfftfreq(len(x), 1/fs)

    dominant_frequency = frequencies[np.argmax(spectrum)]

    spectral_centroid = (
        np.sum(frequencies * spectrum) /
        (np.sum(spectrum) + 1e-12)
    )

    return [
        rms,
        peak,
        kurtosis(x),
        crest,
        skew(x),
        dominant_frequency,
        spectral_centroid
    ]


# ---------------- RESPONSE SIMULATION ----------------

def emergency_shutdown(severity):
    if severity == "Critical":
        return {
            "shutdown": True,
            "alarm": True,
            "status": "SIMULATED EMERGENCY SHUTDOWN"
        }

    elif severity == "Warning":
        return {
            "shutdown": False,
            "alarm": True,
            "status": "MAINTENANCE WARNING"
        }

    return {
        "shutdown": False,
        "alarm": False,
        "status": "NORMAL OPERATION"
    }


# ---------------- PAGE HEADER ----------------

st.title("🛡️ Smart Machine Guardian")
st.markdown(
    "### Vibration-Based Fault Detection and Predictive Maintenance"
)

st.caption(
    "AI-powered bearing condition monitoring | CWRU dataset | "
    "12 kHz sampling rate"
)

st.divider()

# ---------------- SIDEBAR ----------------

st.sidebar.title("System Information")
st.sidebar.success("AI Models Loaded")

st.sidebar.write("**Classification:** SVM")
st.sidebar.write(f"**Sampling rate:** {FS} Hz")
st.sidebar.write(f"**Segment size:** {SEGMENT_SIZE}")
st.sidebar.write("**Filter:** Saved Butterworth bandpass")

st.sidebar.info(
    "Prototype validated using CWRU Drive End vibration recordings. "
    "Use compatible 12 kHz signals for meaningful predictions."
)

# ---------------- FILE UPLOAD ----------------

st.subheader("1. Upload Vibration Recording")

uploaded_file = st.file_uploader(
    "Upload a CWRU MATLAB (.mat) vibration file",
    type=["mat"]
)

if uploaded_file is not None:

    try:
        mat_data = scipy.io.loadmat(uploaded_file)

        de_keys = [
            key for key in mat_data.keys()
            if "_DE_time" in key and not key.startswith("__")
        ]

        if not de_keys:
            st.error(
                "No Drive End (_DE_time) signal was found in this MAT file."
            )
            st.stop()

        selected_key = st.selectbox(
            "Select Drive End vibration signal",
            de_keys
        )

        raw_signal = np.asarray(
            mat_data[selected_key]
        ).squeeze().astype(float)

        if raw_signal.ndim != 1 or len(raw_signal) < SEGMENT_SIZE:
            st.error("The selected signal is too short or invalid.")
            st.stop()

        st.success(
            f"Loaded {selected_key} successfully — "
            f"{len(raw_signal):,} samples"
        )

        st.write(
            f"**Recording duration:** "
            f"{len(raw_signal)/FS:.2f} seconds"
        )

        # ---------------- SIGNAL PROCESSING ----------------

        filtered_signal = sig.sosfiltfilt(SOS, raw_signal)

        # ---------------- WAVEFORM GRAPHS ----------------

        st.subheader("2. Signal Analysis")

        display_length = min(4000, len(raw_signal))
        time_axis = np.arange(display_length) / FS

        fig = make_subplots(
            rows=2,
            cols=1,
            subplot_titles=(
                "Raw Vibration Signal",
                "Bandpass-Filtered Vibration Signal"
            )
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=raw_signal[:display_length],
                name="Raw signal"
            ),
            row=1,
            col=1
        )

        fig.add_trace(
            go.Scatter(
                x=time_axis,
                y=filtered_signal[:display_length],
                name="Filtered signal"
            ),
            row=2,
            col=1
        )

        fig.update_layout(
            height=600,
            showlegend=False,
            xaxis2_title="Time (seconds)",
            yaxis_title="Amplitude",
            yaxis2_title="Amplitude"
        )

        st.plotly_chart(fig, use_container_width=True)

        # ---------------- FFT ----------------

        st.subheader("3. Frequency Spectrum")

        fft_segment = filtered_signal[:SEGMENT_SIZE]
        fft_values = np.abs(np.fft.rfft(fft_segment))
        fft_freq = np.fft.rfftfreq(SEGMENT_SIZE, 1/FS)

        fft_fig = go.Figure()

        fft_fig.add_trace(
            go.Scatter(
                x=fft_freq,
                y=fft_values,
                mode="lines",
                name="FFT"
            )
        )

        fft_fig.update_layout(
            title="FFT of Filtered Vibration Signal",
            xaxis_title="Frequency (Hz)",
            yaxis_title="Magnitude",
            height=400
        )

        st.plotly_chart(fft_fig, use_container_width=True)

        # ---------------- AI PREDICTION ----------------

        st.subheader("4. AI Fault Diagnosis")

        if st.button("Analyze Vibration Signal", type="primary"):

            feature_rows = []
            segment_numbers = []

            number_of_segments = len(filtered_signal) // SEGMENT_SIZE

            for i in range(number_of_segments):

                start = i * SEGMENT_SIZE
                end = start + SEGMENT_SIZE

                segment = filtered_signal[start:end]

                features = extract_features(segment, FS)

                feature_rows.append(features)
                segment_numbers.append(i + 1)

            feature_df = pd.DataFrame(
                feature_rows,
                columns=FEATURES
            )

            severity_predictions = severity_model.predict(feature_df)
            fault_predictions = fault_model.predict(feature_df)

            results = pd.DataFrame({
                "Segment": segment_numbers,
                "Predicted Severity": severity_predictions,
                "Predicted Fault": fault_predictions
            })

            final_severity = pd.Series(
                severity_predictions
            ).mode()[0]

            final_fault = pd.Series(
                fault_predictions
            ).mode()[0]

            response = emergency_shutdown(final_severity)

            st.session_state["results"] = results
            st.session_state["severity"] = final_severity
            st.session_state["fault"] = final_fault
            st.session_state["response"] = response

        # ---------------- DISPLAY RESULTS ----------------

        if "results" in st.session_state:

            results = st.session_state["results"]
            severity = st.session_state["severity"]
            fault = st.session_state["fault"]
            response = st.session_state["response"]

            st.divider()
            st.subheader("5. Diagnosis Results")

            col1, col2, col3 = st.columns(3)

            col1.metric("Severity", severity)
            col2.metric("Fault Type", fault)
            col3.metric(
                "Analyzed Segments",
                len(results)
            )

            if severity == "Healthy":
                st.success("SYSTEM HEALTHY — NORMAL OPERATION")

            elif severity == "Warning":
                st.warning("MAINTENANCE REQUIRED — WARNING DETECTED")

            else:
                st.error("CRITICAL CONDITION DETECTED")

            # ---------------- SHUTDOWN PANEL ----------------

            st.subheader("6. Automatic Response System")

            if response["shutdown"]:
                st.error("🚨 " + response["status"])
            elif response["alarm"]:
                st.warning("⚠️ " + response["status"])
            else:
                st.success("✅ " + response["status"])

            shutdown_col, alarm_col = st.columns(2)

            shutdown_col.metric(
                "Simulated Shutdown",
                "ACTIVE" if response["shutdown"] else "INACTIVE"
            )

            alarm_col.metric(
                "Alarm",
                "ACTIVE" if response["alarm"] else "INACTIVE"
            )

            st.caption(
                "Demonstration only: shutdown and alarm are software "
                "simulation outputs. No physical machinery is controlled."
            )

            # ---------------- SEGMENT TABLE ----------------

            st.subheader("7. Segment-by-Segment Results")

            st.dataframe(
                results,
                use_container_width=True
            )

            csv_data = results.to_csv(index=False)

            st.download_button(
                "Download Prediction Report (CSV)",
                data=csv_data,
                file_name="smart_machine_guardian_results.csv",
                mime="text/csv"
            )

    except Exception as e:
        st.error(f"Analysis error: {e}")

else:
    st.info(
        "Upload a compatible CWRU .mat file to begin vibration analysis."
    )

# ---------------- FOOTER ----------------

st.divider()

st.caption(
    "Smart Machine Guardian | Signals & Systems CEP | "
    "Research prototype — not certified for industrial safety use."
)
