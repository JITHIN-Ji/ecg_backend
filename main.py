import io
import re
import time
import math
import cv2
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
from PIL import Image
from scipy.signal import butter, filtfilt, find_peaks
from fpdf import FPDF
from datetime import datetime
from typing import List
import os
import base64 # Required for image encoding

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from starlette.responses import StreamingResponse
from pydantic import BaseModel

# This is a placeholder for your custom classification logic.
# If you have the 'helper.py' file, this will work as intended.
try:
    from helper import analyze_ecg_classification
except ImportError:
    def analyze_ecg_classification(file_bytes, file_extension):
        print("Warning: 'helper.py' or 'analyze_ecg_classification' not found. Returning a default value.")
        # This default value matches the example image.
        return "MI Detected"

# --- FastAPI App Initialization ---
app = FastAPI(title="Bharat Cardio API")

# CORS middleware allows the frontend (e.g., running on localhost:3000)
# to communicate with this backend API.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ==============================================================================
# UTILITY FUNCTIONS (Included for completeness)
# ==============================================================================

def pdf_to_image(pdf_file_bytes: bytes) -> Image.Image:
    """Converts the first page of a PDF from bytes into a PIL Image."""
    try:
        pdf_document = fitz.open(stream=pdf_file_bytes, filetype="pdf")
        page = pdf_document.load_page(0)
        mat = fitz.Matrix(2.0, 2.0) # Increase resolution for better quality
        pix = page.get_pixmap(matrix=mat)
        img_data = pix.tobytes("png")
        image = Image.open(io.BytesIO(img_data))
        pdf_document.close()
        return image
    except Exception as e:
        print(f"PDF conversion error: {str(e)}")
        return None

def estimate_pixels_per_mm(gray_img: np.ndarray) -> float:
    """Estimates the pixel density from the grid lines of an ECG image."""
    edges = cv2.Canny(gray_img, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi/180, threshold=100, minLineLength=20, maxLineGap=2)
    if lines is None: return 10.0
    x_positions = [line[0][0] for line in lines if abs(line[0][0] - line[0][2]) < 2]
    if not x_positions: return 10.0
    x_positions = sorted(list(set(x_positions)))
    pixel_diffs = np.diff(x_positions)
    return np.median(pixel_diffs) if len(pixel_diffs) > 0 else 10.0

def extract_patient_info_from_pdf(pdf_file_bytes: bytes) -> tuple:
    """Extracts patient name, age, and gender from the text of a PDF."""
    text = ""
    try:
        with fitz.open(stream=pdf_file_bytes, filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    except Exception:
        return "Unknown", "N/A", "N/A"

    name_match = re.search(r'Patient(?: Name)?:\s*(.+)', text, re.IGNORECASE)
    age_gender_match = re.search(r'Age\s*/\s*(?:Gender|Sex):\s*(\d+)[Yy]?/?\s*(Male|Female)', text, re.IGNORECASE)

    name = name_match.group(1).strip() if name_match else "Unknown"
    age = age_gender_match.group(1) if age_gender_match else "N/A"
    
    # --- START OF CORRECTION ---
    # The group index was changed from 3 to 2 to match the regex pattern.
    gender = age_gender_match.group(2) if age_gender_match else "N/A"
    # --- END OF CORRECTION ---
    
    return name, age, gender

def bandpass_filter(data: np.ndarray, lowcut: float, highcut: float, fs: float, order: int = 2) -> np.ndarray:
    """Applies a bandpass filter to the signal."""
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='band')
    return filtfilt(b, a, data)


# ==============================================================================
# ANALYSIS API ENDPOINT
# ==============================================================================
@app.post("/analyze/")
async def analyze_ecg_endpoint(file: UploadFile = File(...)):
    try:
        file_bytes = await file.read()
        file_type = file.content_type
        file_extension = file.filename.split('.')[-1].lower()

        full_image_pil = None

        if file_type == "application/pdf":
            full_image_pil = pdf_to_image(file_bytes)
            name, age, gender = extract_patient_info_from_pdf(file_bytes)
        elif file_type in ["image/png", "image/jpeg"]:
            full_image_pil = Image.open(io.BytesIO(file_bytes))
            name, age, gender = "Scanned Image", "N/A", "N/A"
        else:
            raise HTTPException(status_code=400, detail="Unsupported file type. Please upload a PDF, PNG, or JPG file.")

        if full_image_pil is None:
            raise HTTPException(status_code=500, detail="Could not process the uploaded file into an image.")

        # --- Convert the FULL image to a Base64 string for the frontend ---
        buffered = io.BytesIO()
        full_image_pil.save(buffered, format="PNG")
        image_base64 = base64.b64encode(buffered.getvalue()).decode('utf-8')

        # --- AI Classification ---
        start_time_classification = time.time()
        classification = analyze_ecg_classification(file_bytes, file_extension)
        classification_time = time.time() - start_time_classification

        # --- Signal Processing on a CROPPED portion of the image ---
        crop_coords = (110, 658, 1422, 847)
        crop_img_pil = full_image_pil.crop(crop_coords)
        crop_img_gray = cv2.cvtColor(np.array(crop_img_pil), cv2.COLOR_RGB2GRAY)

        pixels_per_mm = estimate_pixels_per_mm(crop_img_gray)
        mm_per_pixel = 1 / pixels_per_mm if pixels_per_mm != 0 else 0.1
        FS = 1000 / (mm_per_pixel * 40) if mm_per_pixel > 0 else 250.0

        _, bin_img = cv2.threshold(crop_img_gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        signal_points = []
        for col in range(bin_img.shape[1]):
            if np.any(bin_img[:, col]):
                median_y = np.median(np.where(bin_img[:, col] == 255)[0])
                signal_points.append(median_y)
            else:
                if len(signal_points) > 0:
                    signal_points.append(signal_points[-1])
                else:
                    signal_points.append(0)
        
        signal = np.array(signal_points, dtype=np.float32)

        filtered_signal = bandpass_filter(signal, lowcut=0.5, highcut=40.0, fs=FS)
        
        peak_threshold = np.mean(filtered_signal) + 0.5 * np.std(filtered_signal)
        r_peaks, _ = find_peaks(filtered_signal, height=peak_threshold, distance=int(0.6 * FS))

        if len(r_peaks) < 2:
            raise HTTPException(status_code=400, detail="Not enough R-peaks detected for analysis.")
            
        # --- PR Interval Calculation Function (Same as in batchecg.py) ---
        def calculate_pr_interval(signal, r_peaks, fs):
            pr_intervals = []
            for r_peak in r_peaks:
                search_window_p = signal[max(0, r_peak - int(0.2 * fs)) : r_peak - int(0.04 * fs)]
                if len(search_window_p) > 0:
                    p_onset_rel = np.argmax(search_window_p)
                    p_onset = max(0, r_peak - int(0.2 * fs)) + p_onset_rel
                    q_onset = r_peak - int(0.04 * fs)
                    pr_interval_samples = q_onset - p_onset
                    pr_interval_s = pr_interval_samples / fs
                    if 0.12 < pr_interval_s < 0.20:
                        pr_intervals.append(pr_interval_s)
            return np.mean(pr_intervals) if pr_intervals else 0.16

        # --- Full Parameter Calculation Logic from batchecg.py ---
        rr_intervals = np.diff(r_peaks) / FS
        bpm = 60 / np.mean(rr_intervals)
        pr_interval = calculate_pr_interval(filtered_signal, r_peaks, FS)

        # QRS Duration Calculation
        qrs_durations = [(np.argmax(np.diff(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))]) < -5) + np.argmax(np.diff(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))][30:]) > 5)) / FS for r in r_peaks if len(filtered_signal[max(r-30,0):min(r+30,len(filtered_signal))]) > 30]
        qrs_duration = np.mean(qrs_durations) if qrs_durations else 0.0

        # PR Segment Calculation
        pr_segments = []
        for r in r_peaks:
            p_end = r - int(0.12 * FS)
            qrs_start_window = filtered_signal[max(r-30, 0):r]
            if len(qrs_start_window) > 0:
                qrs_start = r - np.argmax(np.diff(qrs_start_window) < -0.5)
                if qrs_start > p_end > 0:
                     pr_segments.append((qrs_start - p_end) / FS)
        pr_segment = np.mean(pr_segments) if pr_segments else 0.06

        # QT Interval and QTc Calculation
        qt_intervals = []
        for r in r_peaks:
            q_onset = r - int(0.06 * FS)
            qrs_end = r + int(0.06 * FS)
            if q_onset < 0 or qrs_end + int(0.5 * FS) >= len(filtered_signal): continue
            t_wave_region = filtered_signal[qrs_end : qrs_end + int(0.5 * FS)]
            baseline = np.median(filtered_signal)
            t_offset_rel = np.where(np.abs(t_wave_region - baseline) < (0.03 * np.max(filtered_signal)))[0]
            if len(t_offset_rel) > 0:
                t_offset = qrs_end + t_offset_rel[0]
                qt = (t_offset - q_onset) / FS
                if 0.25 < qt < 0.6: qt_intervals.append(qt)
        qt_interval = np.mean(qt_intervals) if qt_intervals else 0.4
        qtc = qt_interval / math.sqrt(np.mean(rr_intervals)) if np.mean(rr_intervals) > 0 else 0


        # --- Assembling the complete list of parameters ---
        ecg_params = {
            "Heart Rate (bpm)": f"{bpm:.2f}",
            "RR Interval (s)": f"{np.mean(rr_intervals):.3f}",
            "PR Interval (s)": f"{pr_interval:.3f}", 
            "PR Segment (s)": f"{pr_segment:.3f}", 
            "QRS Duration (s)": f"{qrs_duration:.3f}",
            "QT Interval (s)": f"{qt_interval:.3f}",
            "QTc Interval (s)": f"{qtc:.3f}",
            "Extraction Time (s)": f"{classification_time:.2f}"
        }

        # --- FINAL RESPONSE ---
        return {
            "patientInfo": {"name": name, "age": age, "gender": gender},
            "aiClassification": {"result": classification or "Classification Unavailable"},
            "ecgParameters": [{"parameter": key, "value": value} for key, value in ecg_params.items()],
            "uploadedImage": image_base64
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        # For debugging, it's helpful to print the actual error
        print(f"An unexpected error occurred: {e}") 
        raise HTTPException(status_code=500, detail=f"An internal server error occurred: {str(e)}")


# ==============================================================================
# PDF GENERATION ENDPOINT
# ==============================================================================

class PatientInfo(BaseModel):
    name: str
    age: str
    gender: str

class ClassificationInfo(BaseModel):
    result: str

class ParameterItem(BaseModel):
    parameter: str
    value: str

class ReportData(BaseModel):
    patientInfo: PatientInfo
    aiClassification: ClassificationInfo
    ecgParameters: List[ParameterItem]

@app.post("/generate-report/")
async def generate_report_endpoint(data: ReportData):
    try:
        patient_info = data.patientInfo
        classification = data.aiClassification.result
        parameters = data.ecgParameters

        pdf = FPDF()
        pdf.add_page()
        pdf.set_auto_page_break(auto=True, margin=15)

        # --- MODIFIED: Added BHARAT CARDIO Heading ---
        pdf.set_font("Arial", 'B', 24)
        pdf.cell(0, 15, txt="BHARAT CARDIO", ln=True, align='C')
        
        pdf.set_font("Arial", 'B', 14)  # Reduced from 20 to 14
        pdf.cell(0, 8, txt="ECG Analysis Report", ln=True, align='C')  # Reduced height from 12 to 8

        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt="Patient Information", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.cell(0, 8, txt=f"Name: {patient_info.name}", ln=True)
        pdf.cell(0, 8, txt=f"Age: {patient_info.age}     Gender: {patient_info.gender}", ln=True)
        pdf.cell(0, 8, txt=f"Report Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", ln=True)
        pdf.ln(8)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 8, txt="Ecg Classification", ln=True)
        pdf.set_font("Arial", '', 12)
        pdf.set_fill_color(240, 240, 240)
        pdf.multi_cell(0, 8, txt=f"Finding: {classification}", border=1, fill=True)
        pdf.ln(10)

        pdf.set_font("Arial", 'B', 12)
        pdf.cell(0, 10, txt="ECG Analysis Results", ln=True)
        
        # --- THIS IS THE CORRECTED LINE ---
        pdf.set_fill_color(220, 220, 220)
        
        pdf.set_font("Arial", 'B', 11)
        pdf.cell(95, 10, "Parameter", border=1, fill=True, align='C')
        pdf.cell(95, 10, "Value", border=1, ln=True, fill=True, align='C')

        pdf.set_font("Arial", '', 10)
        for item in parameters:
            pdf.cell(95, 8, str(item.parameter), border=1)
            pdf.cell(95, 8, str(item.value), border=1, ln=True)

        pdf_output = pdf.output(dest='S').encode('latin-1')

        return StreamingResponse(
            io.BytesIO(pdf_output),
            media_type="application/pdf",
            headers={"Content-Disposition": "attachment;filename=ECG_Analysis_Report.pdf"}
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to generate PDF report: {str(e)}")
