import os

class FileCleaner:
    def __init__(self, directory):
        self.directory = directory

    def remove_cleaned_files(self, extensions=[".wav", ".txt"]):
        """
        Supprime tous les fichiers contenant "_cleaned" dans le nom et ayant l'une des extensions spécifiées.
        """
        removed_files = []
        for fname in os.listdir(self.directory):
            if "_cleaned" in fname and any(fname.endswith(ext) for ext in extensions):
                full_path = os.path.join(self.directory, fname)
                try:
                    os.remove(full_path)
                    removed_files.append(fname)
                except Exception as e:
                    print(f"Erreur lors de la suppression de {fname} : {e}")

        print(f"{len(removed_files)} fichiers supprimés :")
        for f in removed_files:
            print(f" - {f}")

