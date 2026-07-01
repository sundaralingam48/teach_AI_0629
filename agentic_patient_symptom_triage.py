import os
from datetime import datetime
import google.generativeai as genai
from google.colab import userdata

# Configure Gemini API
genai.configure(api_key=userdata.get('GoogleAPIKey'))

class HealthcareAgentPipeline:
    def __init__(self, model_name='gemini-2.5-flash'):
        """Initializes the pipeline with the specified Gemini model."""
        self.model = genai.GenerativeModel(
            model_name, 
            generation_config={"temperature": 0.3}
        )

    def symptom_triage_agent(self, user_input: str) -> str:
        """Agent 1: Analyzes symptoms and assesses risk/urgency level."""
        print("\n[1. Symptom Triage Agent]")
        prompt = f"""
        You are an expert medical triage assistant. Based on the user's reported symptoms, 
        classify the urgency level (Low, Medium, High). 
        List the top medical concerns or red flags in 2-3 lines.
        Do not provide a definitive medical diagnosis.
        
        User Symptoms: {user_input}
        """
        response = self.model.generate_content(prompt)
        print(response.text)
        return response.text

    def treatment_recommender_agent(self, triage_summary: str) -> str:
        """Agent 2: Provides general guidance based on triage severity."""
        print("\n[2. Treatment Recommender Agent]")
        prompt = f"""
        You are a medical consultation assistant. Based on the triage summary, 
        suggest the top 3 safest next steps or general home-care advice (if applicable).
        Keep the output confined to 2-3 lines and in bullet points. Always include a disclaimer 
        to seek professional care if condition worsens.
        
        Triage Summary: {triage_summary}
        """
        response = self.model.generate_content(prompt)
        print(response.text)
        return response.text

    def clinical_planning_agent(self, triage_summary: str, recommendation: str) -> str:
        """Agent 3: Formulates smart follow-up questions for the doctor's visit."""
        print("\n[3. Clinical Planning Agent]")
        prompt = f"""
        You are a clinical intake assistant. Based on the triage summary and initial recommendations, 
        what specific medical history or diagnostic follow-up questions should the doctor ask the patient 
        during their visit to get clarity? Keep the output to 2-3 lines and in bullet points.
        
        Triage Summary: {triage_summary}
        Recommendation: {recommendation}
        """
        response = self.model.generate_content(prompt)
        print(response.text)
        return response.text

    def logger_agent(self, name: str, patient_id: str, input_text: str, triage: str, recs: str, followups: str):
        """Agent 4: Logs the entire interaction safely into a file."""
        print("\n[4. Logger Agent]")
        
        log_entry = f"""
        ============================================================
        Timestamp: {datetime.now()}
        Patient Name: {name}
        Patient ID: {patient_id}
        ------------------------------------------------------------
        Patient Reported Symptoms: 
        {input_text}
        
        Triage Assessment: 
        {triage.strip()}
        
        Care Recommendations: 
        {recs.strip()}
        
        Suggested Clinical Follow-ups: 
        {followups.strip()}
        ============================================================\n
        """
        
        with open("healthcare_triage_log.txt", "a") as f:
            f.write(log_entry)
        print("Successfully saved clinical intake log to 'healthcare_triage_log.txt'.")

    def run_pipeline(self):
        """Orchestrates the entire multi-agent workflow."""
        print("--- Healthcare Intake System ---")
        name = input("Enter patient name: ")
        patient_id = input("Enter Patient ID: ")
        user_input = input("Please describe your current symptoms (onset, severity, details): ")

        # Step 1: Triage
        triage_summary = self.symptom_triage_agent(user_input)

        # Step 2: Recommendations
        recommendation = self.treatment_recommender_agent(triage_summary)

        # Step 3: Follow-up Questions
        followups = self.clinical_planning_agent(triage_summary, recommendation)

        # Step 4: Log Data
        self.logger_agent(name, patient_id, user_input, triage_summary, recommendation, followups)


# --- Execution ---
if __name__ == "__main__":
    # Instantiate and run the pipeline
    pipeline = HealthcareAgentPipeline()
    pipeline.run_pipeline()
