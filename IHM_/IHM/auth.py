from sqlalchemy.orm import Session
from IHM_.IHM.models import Medecin

def authenticate_user(db: Session, email: str, password: str):
    medecin = db.query(Medecin).filter(Medecin.email == email).first()
    if medecin and medecin.mot_de_passe == password:
        return medecin
    return None
