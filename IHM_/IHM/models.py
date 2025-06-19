from sqlalchemy import Column, Integer, String
from IHM_.IHM.database import Base


class Medecin(Base):
    __tablename__ = "medecins"

    id = Column(Integer, primary_key=True, index=True)
    nom = Column(String(100))
    email = Column(String(100), unique=True, index=True)
    mot_de_passe = Column(String(100))
