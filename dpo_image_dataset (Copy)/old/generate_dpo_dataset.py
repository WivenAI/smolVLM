#!/usr/bin/env python3
"""
Système d'agents CrewAI pour la génération de jeux de données DPO
Génère un jeu de données DPO au format texte + image vers texte
"""

import os
import json
import base64
from pathlib import Path
from typing import List, Dict
import google.generativeai as genai
from crewai import Agent, Task, Crew, Process, LLM

# Configure Gemini API
with open('/home/david-lacour/Desktop/geminiAPIkey.txt', 'r') as f:
    GEMINI_API_KEY = f.read().strip()

genai.configure(api_key=GEMINI_API_KEY)

# Configure LLM for CrewAI
llm = LLM(
    model="gemini/gemini-2.5-flash",
    api_key=GEMINI_API_KEY
)

# Working directory
WORKING_DIR = "/home/david-lacour/Documents/Wiven/data/exctractWithContext/gemini/output_same_folder_without_icone"


def load_image_base64(image_path: str) -> str:
    """Charge une image et la convertit en base64"""
    with open(image_path, 'rb') as f:
        return base64.b64encode(f.read()).decode('utf-8')


def read_context_file(text_path: str) -> str:
    """Lit le fichier texte de contexte"""
    with open(text_path, 'r', encoding='utf-8') as f:
        return f.read()


def get_image_text_pairs() -> List[Dict[str, str]]:
    """Récupère toutes les paires image-texte du répertoire de travail"""
    pairs = []
    working_path = Path(WORKING_DIR)

    # Trouve tous les fichiers PNG
    for image_file in sorted(working_path.glob("image_*.png")):
        # Récupère le fichier texte correspondant
        text_file = image_file.with_suffix('.txt')

        if text_file.exists():
            pairs.append({
                'image_path': str(image_file),
                'text_path': str(text_file),
                'image_name': image_file.name
            })

    return pairs


class GeminiVisionAnalyzer:
    """Wrapper pour l'API Gemini Vision"""

    def __init__(self):
        self.model = genai.GenerativeModel('gemini-2.5-flash')

    def analyze_image_with_context(self, image_path: str, context: str) -> str:
        """Analyse une image avec le contexte fourni"""
        from PIL import Image

        # Charge l'image
        image = Image.open(image_path)

        prompt = f"""Vous analysez une image provenant d'un système de documentation logicielle.

Contexte de la documentation :
{context}

Veuillez analyser cette image et fournir :
1. Une description détaillée de ce que montre l'image
2. Comment elle se rapporte au contexte fourni
3. Quels éléments d'interface utilisateur, boutons ou fonctionnalités spécifiques sont visibles
4. Tout texte ou étiquette visible dans l'image

Soyez précis et technique dans votre analyse."""

        response = self.model.generate_content([prompt, image])
        return response.text


# Créer les agents CrewAI
image_analysis_agent = Agent(
    role='Spécialiste en analyse d\'images',
    goal='Analyser les images de documentation logicielle et extraire des informations visuelles détaillées',
    backstory="""Vous êtes un expert dans l'analyse de captures d'écran d'interface utilisateur et d'images de documentation.
    Vous avez une grande expérience dans l'identification d'éléments d'interface, la compréhension des flux de travail logiciels,
    et la description d'interfaces techniques avec précision.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

prompt_generation_agent = Agent(
    role='Ingénieur en prompts de données d\'entraînement',
    goal='Générer des prompts de haute qualité pour la création de jeux de données DPO',
    backstory="""Vous êtes un expert dans la création de jeux de données d'entraînement pour les modèles vision-langage.
    Vous excellez dans la création de prompts diversifiés et réalistes que les utilisateurs pourraient poser
    lors d'interactions avec un assistant IA visuel sur la documentation logicielle.""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)

response_generation_agent = Agent(
    role='Générateur de réponses d\'assistant',
    goal='Générer des réponses choisies (haute qualité) et rejetées (qualité inférieure) pour l\'entraînement DPO',
    backstory="""Vous êtes un expert dans la génération de paires de réponses pour l'apprentissage par préférence.
    Vous comprenez ce qui fait une bonne réponse d'assistant (précise, utile, bien structurée)
    par opposition à une mauvaise (vague, incomplète ou incorrecte).""",
    verbose=True,
    allow_delegation=False,
    llm=llm
)


def create_dpo_entry(image_path: str, text_path: str, image_name: str) -> Dict:
    """Crée une entrée de jeu de données DPO pour une paire image-texte"""

    # Lit le contexte
    context = read_context_file(text_path)

    # Initialise l'analyseur Gemini
    analyzer = GeminiVisionAnalyzer()

    # Analyse l'image
    image_analysis = analyzer.analyze_image_with_context(image_path, context)

    # Tâche 1 : Générer un prompt descriptif
    task1 = Task(
        description=f"""En vous basant sur cette analyse d'image et ce contexte, générez un prompt descriptif
        qui demande à l'IA de décrire ce qui se trouve dans l'image.

        Analyse de l'image : {image_analysis[:500]}...
        Contexte : {context[:500]}...

        Générez un prompt utilisateur naturel demandant des informations sur le contenu de l'image.""",
        agent=prompt_generation_agent,
        expected_output="Un prompt en langage naturel demandant des informations sur l'image"
    )

    # Tâche 2 : Générer un prompt de Q&R
    task2 = Task(
        description=f"""En vous basant sur cette analyse d'image et ce contexte, générez une question qu'un utilisateur
        pourrait poser sur cette fonctionnalité logicielle ou cet élément d'interface montré dans l'image.

        Analyse de l'image : {image_analysis[:500]}...
        Contexte : {context[:500]}...

        Générez une question utilisateur réaliste sur la fonctionnalité montrée.""",
        agent=prompt_generation_agent,
        expected_output="Une question utilisateur réaliste sur la fonctionnalité logicielle"
    )

    # Tâche 3 : Générer une réponse choisie (haute qualité)
    task3 = Task(
        description=f"""Générez une réponse d'assistant de haute qualité qui :
        1. Décrit précisément l'image
        2. Fait référence au contexte pertinent
        3. Est utile, claire et bien structurée

        Analyse de l'image : {image_analysis}
        Contexte : {context}

        Ceci doit être la réponse 'choisie' - la réponse préférée, de haute qualité.""",
        agent=response_generation_agent,
        expected_output="Une réponse d'assistant de haute qualité et détaillée"
    )

    # Tâche 4 : Générer une réponse rejetée (qualité inférieure)
    task4 = Task(
        description=f"""Générez une réponse d'assistant de qualité inférieure qui :
        1. Est vague ou incomplète
        2. Manque des détails importants de l'image
        3. Ne répond pas complètement à la question

        Analyse de l'image : {image_analysis}
        Contexte : {context}

        Ceci doit être la réponse 'rejetée' - une réponse moins utile.""",
        agent=response_generation_agent,
        expected_output="Une réponse d'assistant de qualité inférieure et vague"
    )

    # Crée l'équipe et exécute les tâches
    crew = Crew(
        agents=[prompt_generation_agent, response_generation_agent],
        tasks=[task1, task2, task3, task4],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    # Analyse les résultats
    task_outputs = [task.output.raw for task in crew.tasks]

    # Crée deux entrées DPO par image
    entries = [
        {
            "prompt": task_outputs[0],  # Prompt descriptif
            "chosen": task_outputs[2],   # Réponse de haute qualité
            "rejected": task_outputs[3], # Réponse de qualité inférieure
            "image_name": image_name,
            "type": "descriptive"
        },
        {
            "prompt": task_outputs[1],  # Prompt Q&R
            "chosen": task_outputs[2],   # Réponse de haute qualité
            "rejected": task_outputs[3], # Réponse de qualité inférieure
            "image_name": image_name,
            "type": "qa"
        }
    ]

    return entries


def load_existing_dataset(output_file):
    """Charge le jeu de données existant s'il existe"""
    if os.path.exists(output_file):
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except:
            return []
    return []


def get_processed_images(dataset):
    """Récupère l'ensemble des noms d'images déjà traitées"""
    processed = set()
    for entry in dataset:
        processed.add(entry['image_name'])
    return processed


def save_dataset(output_file, dataset):
    """Sauvegarde le jeu de données dans un fichier"""
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)


def main():
    """Fonction principale d'exécution"""
    print("Démarrage de la génération du jeu de données DPO...")
    print(f"Répertoire de travail : {WORKING_DIR}")

    # Fichier de sortie
    output_file = os.path.join(WORKING_DIR, "dpo_dataset.json")

    # Charge le jeu de données existant (capacité de reprise)
    dpo_dataset = load_existing_dataset(output_file)
    processed_images = get_processed_images(dpo_dataset)

    if processed_images:
        print(f"Reprise depuis l'exécution précédente. Déjà traité : {len(processed_images)} images")
        print(f"Taille actuelle du jeu de données : {len(dpo_dataset)} entrées")

    # Récupère toutes les paires image-texte
    pairs = get_image_text_pairs()
    print(f"Trouvé {len(pairs)} paires image-texte")

    # Filtre les images déjà traitées
    pairs_to_process = [p for p in pairs if p['image_name'] not in processed_images]
    print(f"Images à traiter : {len(pairs_to_process)}")

    # Génère le jeu de données DPO
    for i, pair in enumerate(pairs_to_process, 1):
        print(f"\nTraitement {i}/{len(pairs_to_process)} : {pair['image_name']}")

        try:
            entries = create_dpo_entry(
                pair['image_path'],
                pair['text_path'],
                pair['image_name']
            )
            dpo_dataset.extend(entries)
            print(f"✓ Généré {len(entries)} entrées pour {pair['image_name']}")

            # Sauvegarde la progression immédiatement après chaque image
            save_dataset(output_file, dpo_dataset)
            print(f"💾 Progression sauvegardée ({len(dpo_dataset)} entrées au total)")

        except Exception as e:
            print(f"✗ Erreur lors du traitement de {pair['image_name']} : {e}")
            import traceback
            traceback.print_exc()
            continue

    # Sauvegarde finale (redondant mais sûr)
    save_dataset(output_file, dpo_dataset)

    print(f"\n✓ Jeu de données DPO terminé : {output_file}")
    print(f"Entrées totales : {len(dpo_dataset)}")
    print(f"Images totales traitées : {len(get_processed_images(dpo_dataset))}")


if __name__ == "__main__":
    main()
