document.addEventListener("DOMContentLoaded", () => {
    const uploadForm = document.getElementById("uploadForm");
    const resultContainer = document.getElementById("resultContainer");
    const outputText = document.getElementById("outputText");

    const btnTranscription = document.getElementById("btnTranscription");
    const btnComprehension = document.getElementById("btnComprehension");
    const btnScore = document.getElementById("btnScore");
    const pipelineSteps = document.getElementById("pipelineSteps");

    uploadForm.addEventListener("submit", async (e) => {
        e.preventDefault();
        const fileInput = document.getElementById("audioFile");
        const file = fileInput.files[0];
        if (!file) {
            alert("Veuillez sélectionner un fichier audio (.m4a)");
            return;
        }

        const formData = new FormData();
        formData.append("file", file);

        try {
            const response = await fetch("/upload-audio/", {
                method: "POST",
                body: formData
            });
            const data = await response.json();
            alert(data.status || "Fichier envoyé.");
            pipelineSteps.style.display = "block";
        } catch (error) {
            alert("Une erreur est survenue lors de l'envoi.");
        }
    });

    btnTranscription.addEventListener("click", async () => {
        outputText.textContent = "Transcription & Diarisation en cours...";
        resultContainer.style.display = "block";

        try {
            const response = await fetch("/run-step/?step=transcription", {
                method: "POST"
            });
            const data = await response.json();
            outputText.textContent = data.output || "Erreur : " + data.error;
            btnComprehension.style.display = "inline-block";
        } catch (err) {
            outputText.textContent = "Erreur lors de la transcription.";
        }
    });

    btnComprehension.addEventListener("click", async () => {
        outputText.textContent = "Compréhension en cours...";
        try {
            const response = await fetch("/run-step/?step=comprehension", {
                method: "POST"
            });
            const data = await response.json();
            outputText.textContent = data.output || " Erreur : " + data.error;
            btnScore.style.display = "inline-block";
        } catch (err) {
            outputText.textContent = " Erreur lors de la compréhension.";
        }
    });

    btnScore.addEventListener("click", async () => {
        outputText.textContent = "Calcul du score final...";
        try {
            const response = await fetch("/run-step/?step=score", {
                method: "POST"
            });
            const data = await response.json();
            outputText.textContent = data.output || " Erreur : " + data.error;
        
            actionBtn.style.display = "inline-block";
        
        } catch (err) {
            outputText.textContent = " Erreur lors du calcul du score.";
        }
    });

    const actionBtn = document.getElementById("actionBtn");
    const actionContainer = document.getElementById("actionContainer");
    const actionBox = document.getElementById("actionBox");
    
    
    actionBtn.addEventListener("click", async () => {
        try {
            const response = await fetch("/get-score/");
            const result = await response.json();
            const score = result.score;
    
            let actionText = "";
            let color = "";
            
            if (score > 30) {
                actionText = "Action : Envoyer le SMUR";
                color = "#e53935"; // Rouge
            } else if (score >= 15) {
                actionText = "Action : Envoyer un VSAV";
                color = "#fb8c00"; // Orange
            } else {
                actionText = "Action : Envoyer une Ambulance";
                color = "#43a047"; // Vert
            }
            
            actionBox.textContent = actionText;
            actionBox.style.backgroundColor = color;
            actionBox.style.color = "#fff";
            actionBox.style.boxShadow = "0 4px 10px rgba(0,0,0,0.1)";
            actionBox.style.border = "none";
            
            // Cacher tous les autres blocs et afficher le container
    uploadForm.style.display = "none";
    pipelineSteps.style.display = "none";
    resultContainer.style.display = "none";
    actionBtn.style.display = "none";
    actionContainer.style.display = "block";

            
        } catch (error) {
            actionBox.textContent = "Erreur lors de la récupération du score.";
            actionContainer.style.display = "block";
        }
    });
    


});
