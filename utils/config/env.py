import os
from dotenv import load_dotenv

load_dotenv()

# IBM Quantum credentials
qiskit_api_key = os.getenv("QISKIT_API_KEY")
free_instance_crn = os.getenv("FREE_INSTANCE_CRN")
