#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
TP1 Section 4: Mappage Tonal et Encodage d'Affichage

Ce script:
1. Charge les images XYZ depuis ./images_intermediaires_sec3/*_camera_xyz.tiff
2. Applique l'ajustement de luminosité (À IMPLÉMENTER)
3. Applique le mappage tonal:
   - Linéaire (implémenté)
   - Reinhard (À IMPLÉMENTER)
4. Convertit XYZ vers sRGB linéaire (implémenté)
5. Applique l'OETF sRGB (implémenté)
6. Sauvegarde le JPEG final (implémenté)
7. Analyse les artefacts JPEG (À IMPLÉMENTER)
8. Sauvegarde dans ./images_intermediaires_sec4/

Usage:
    python tp1_sec4.py --input-dir images_intermediaires_sec3 --output-dir images_intermediaires_sec4
"""

import numpy as np
import glob
import os
from PIL import Image

from tp1_io import (
    load_tiff,
    save_tiff16,
    linear_to_srgb,
    xyz_to_linear_srgb,
    quantize_to_8bit,
)
from tp1_rapport import (
    html_document,
    section,
    subsection,
    figure,
    table,
    algorithm_box,
    formula_box,
    save_report,
    comparison_grid,
    create_tonemapping_curves_figure,
    create_tonemapping_comparison_figure,
    create_oetf_comparison_figure,
    create_dynamic_range_figure,
)


# =============================================================================
# Ajustement de Luminosité
# =============================================================================


def adjust_brightness(xyz_image, percentile=99):
    """
    Ajuster la luminosité de l'image en normalisant au percentile donné.

    Mesure le percentile spécifié du canal Y (luminance) et divise
    toute l'image par cette valeur pour normaliser la luminosité.

    Args:
        xyz_image: Image XYZ [H, W, 3]
        percentile: Percentile à utiliser pour la normalisation (défaut: 99)

    Returns:
        Image XYZ avec luminosité ajustée

    Indices:
    1. Extraire le canal Y (luminance): Y = xyz_image[:, :, 1]
    2. Filtrer les valeurs valides (Y > 0)
    3. Calculer le percentile spécifié des valeurs valides
    4. Diviser toute l'image par cette valeur
    5. Retourner l'image ajustée
    """
    Y = xyz_image[:, :, 1]
    
    # Compute the percentile of luminance (excluding zeros/negatives)
    valid_Y = Y[Y > 0]
    if len(valid_Y) == 0:
        print("    Warning: No valid luminance values, skipping brightness adjustment")
        return xyz_image.copy()
    
    percentile_value = np.percentile(valid_Y, percentile)
    
    if percentile_value <= 0:
        print("    Warning: Percentile value <= 0, skipping brightness adjustment")
        return xyz_image.copy()
    
    # Divide the entire image by the percentile value
    adjusted = xyz_image / percentile_value
    
    print(f"    Brightness adjustment: divided by {percentile_value:.6f} (1st percentile)")
    
    return adjusted

    # raise NotImplementedError("Ajustement de luminosité à implémenter")

    # L'ajustement de luminosité est déjà implémenté?

# =============================================================================
# Opérateurs de Mappage Tonal
# =============================================================================


def tonemap_linear(xyz_image):
    """
    Mappage tonal linéaire (identité) - pas de compression.

    Les valeurs > 1 seront clippées lors de la conversion finale.

    Args:
        xyz_image: Image XYZ [H, W, 3]

    Returns:
        Image XYZ (copie)
    """
    return xyz_image.copy()


def tonemap_reinhard(xyz_image):
    """
    Mappage tonal de Reinhard: L_out = L_in / (1 + L_in)

    Appliqué à Y (luminance), X et Z sont mis à l'échelle proportionnellement.

    Référence: "Photographic Tone Reproduction for Digital Images" (2002)

    Args:
        xyz_image: Image XYZ [H, W, 3]

    Returns:
        Image XYZ avec mappage tonal appliqué

    Indices:
    1. Extraire le canal Y (luminance): Y = xyz_image[:, :, 1]
    2. Appliquer la formule: Y_mapped = Y / (1 + Y)
    3. Calculer le ratio: scale = Y_mapped / Y (attention aux divisions par zéro!)
    4. Appliquer ce ratio à X et Z également
    5. Retourner l'image résultante
    """
    Y = xyz_image[:, :, 1]
    Y_mapped = Y / (1.0 + Y)
    scale = np.divide(Y_mapped, Y, out=np.ones_like(Y), where=Y > 1e-9)

    return xyz_image * scale[:, :, np.newaxis]

# =============================================================================
# Sauvegarde d'Images
# =============================================================================


def save_jpeg(img_8bit, filepath, quality=95):
    """
    Sauvegarder une image en JPEG.

    Args:
        img_8bit: Image uint8 [H, W, 3]
        filepath: Chemin de sortie
        quality: Qualité JPEG (1-100, défaut: 95)
    """
    Image.fromarray(img_8bit, mode="RGB").save(filepath, "JPEG", quality=quality)
    print(f"  Saved JPEG: {filepath}")


def save_png(img_8bit, filepath):
    """
    Sauvegarder une image en PNG (sans perte).

    Args:
        img_8bit: Image uint8 [H, W, 3]
        filepath: Chemin de sortie
    """
    Image.fromarray(img_8bit, mode="RGB").save(filepath, "PNG")
    print(f"  Saved PNG: {filepath}")


# =============================================================================
# Analyse de Plage Dynamique
# =============================================================================


def analyze_dynamic_range(image_linear):
    """Analyser l'écrêtage des hautes lumières et l'écrasement des ombres."""
    lum = (
        0.2126 * image_linear[:, :, 0]
        + 0.7152 * image_linear[:, :, 1]
        + 0.0722 * image_linear[:, :, 2]
    )

    highlight_pct = np.sum(lum >= 0.99) / lum.size * 100
    shadow_pct = np.sum(lum <= 0.01) / lum.size * 100

    valid = lum[lum > 0]
    if len(valid) > 0:
        min_lum, max_lum = np.percentile(valid, 1), np.percentile(valid, 99)
        dr_stops = np.log2(max_lum / min_lum) if min_lum > 0 else 0
    else:
        dr_stops = 0

    return {
        "highlight_clipped_percent": highlight_pct,
        "shadow_crushed_percent": shadow_pct,
        "dynamic_range_stops": dr_stops,
        "min_luminance": float(np.min(lum)),
        "max_luminance": float(np.max(lum)),
        "mean_luminance": float(np.mean(lum)),
    }


# =============================================================================
# Génération du Rapport HTML
# =============================================================================


def generate_report(results, output_dir):
    """
    Générer un rapport HTML template pour toutes les sections du TP1.
    
    Crée un rapport complet avec:
    - Section 1: Chargement et compréhension des données RAW
    - Section 2: Dématriçage (Demosaicking)
    - Section 3: Balance des Blancs (White Balance)
    - Section 4: Mappage tonal et encodage d'affichage
    
    Inclut toutes les figures générées et des espaces "À remplir" pour l'étudiant.
    """
    # Définir les répertoires de sortie pour chaque section
    # Si output_dir est "images_intermediaires_sec4", base_dir sera le répertoire parent
    if "images_intermediaires_sec" in os.path.basename(output_dir):
        base_dir = os.path.dirname(output_dir) or "."
    else:
        base_dir = output_dir
    
    sec1_dir = os.path.join(base_dir, "images_intermediaires_sec1")
    sec2_dir = os.path.join(base_dir, "images_intermediaires_sec2")
    sec3_dir = os.path.join(base_dir, "images_intermediaires_sec3")
    sec4_dir = output_dir
    
    # Obtenir la liste des basenames (noms de fichiers sans extension)
    basenames = [result["basename"] for result in results] if results else []
    
    # Si aucun résultat, chercher les fichiers dans les répertoires
    if not basenames:
        # Chercher dans sec1
        tiff_files = glob.glob(os.path.join(sec1_dir, "*.tiff"))
        basenames = [os.path.splitext(os.path.basename(f))[0] for f in tiff_files if "zoom" not in f]
        basenames = list(set(basenames))  # Dédupliquer
    
    # Limiter à 2 images d'exemple pour rendre le rapport plus court
    basenames = sorted(basenames)[:3]
    content = ""
    
    # =============================================================================
    # SECTION 1: Chargement et Compréhension des Données RAW
    # =============================================================================
    sec1_content = ""
    
    # Texte d'introduction pour la section 1
    sec1_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4fc3f7;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        '''Le format RAW contient les données brutes du capteur sans traitement. 
        Ces données sont enregistrées en motif de Bayer, où chaque pixel ne capture qu'une seule couleur (R, G ou B). 
        J'ai observé que les images brutes apparaissent très sombres car elles sont linéaires, le capteur compte les photons proportionnellement, et n'ont pas encore subi de correction. 
        La normalisation est essentielle pour ramener les valeurs (dans mon cas sur 12, 14 et 16 bits) vers un intervalle [0, 1]. 
        On soustrait aussi un petit niveau de noir du capteur pour contrer le bruit.'''
        '</p></div>'
    )
    
    for basename in basenames:
        sec1_img_content = ""
        
        # Figure: Zoom sur la mosaïque Bayer
        zoom_path = os.path.join(sec1_dir, f"{basename}_zoom16x16.png")
        if os.path.exists(zoom_path):
            sec1_img_content += subsection(
                f"Région 16×16 de la mosaïque - {basename}",
                figure(f"../images_intermediaires_sec1/{basename}_zoom16x16.png",
                       "Zoom sur une région 16×16 montrant les valeurs normalisées et le motif de Bayer coloré.")
            )
        
        if sec1_img_content:
            sec1_content += section(f"Image: {basename}", sec1_img_content)
    
    # Analyse et observations
    sec1_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #4fc3f7;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        """
                J'ai remarqué que mes images personnelles ont 14 bits de profondeur.
                Celles fourni varie entre 12, 14 et même 16 bits pour le pelican. 
                On comprend vite l'importance de la normalisation pour ramener
                toutes ces valeurs dans un seul intervalle pour le traitement.
            
                Aussi, en zoomant sur le 16x16, on peut voir le filtre de Bayer de l'image.
                Les filtre change d'image en image.
                L'image n'est pas encore dématricée donc le filtre est très apparent quand on zoom.
        
                Finalement, le niveau de noir n'est jamais à 0 dans les métadonnées.
                J'assume que c'est le capteur qui capture de la lumière parasite.
                Le code soustrait on offset avant la normalisation pour ça.
        """
        '</p></div>'
    )
    
    content += section("Section 1: Chargement et Compréhension des Données RAW", sec1_content, icon="📷")
    
    # =============================================================================
    # SECTION 2: Dématriçage (Demosaicking)
    # =============================================================================
    sec2_content = ""
    
    # Texte d'introduction pour la section 2
    sec2_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        "Le dématriçage consiste à interpoler les couleurs manquantes pour obtenir "
        "une image RGB complète. La méthode bilinéaire est rapide mais elle donne des artefacts "
        "de couleur sur les contours (visible quand on zoom). "
        "La méthode Malvar améliore le résultat en utilisant la luminance pour "
        "corriger l'interpolation des canaux rouge et bleu. "
        "Visuellement, on voit donc des contours plus nets et moins d'artefacts de fausses couleurs."
        '</p>'
        '</div>'
    )
    
    for basename in basenames:
        sec2_img_content = ""
        
        # Figure: Comparaison des méthodes
        comp_path = os.path.join(sec2_dir, f"{basename}_comparison.png")
        if os.path.exists(comp_path):
            sec2_img_content += subsection(
                f"Comparaison des méthodes - {basename}",
                figure(f"../images_intermediaires_sec2/{basename}_comparison.png",
                       "Comparaison des méthodes de dématriçage")
            )
        
        # Figure: Zoom sur les artefacts
        zoom_path = os.path.join(sec2_dir, f"{basename}_zoom.png")
        if os.path.exists(zoom_path):
            sec2_img_content += subsection(
                f"Zoom sur les artefacts - {basename}",
                figure(f"../images_intermediaires_sec2/{basename}_zoom.png",
                       "Recadrages montrant les artefacts de contour")
            )
        
        if sec2_img_content:
            sec2_content += section(f"Image: {basename}", sec2_img_content)
    
    # Analyse et observations
    sec2_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        """
            Théoriquement, l'interpolation bilinéaire devrait présenter des artefacts visibles 
            sur les contours nets. Cela est dû au fait qu'elle traite chaque canal indépendamment 
            sans tenir compte de ce que les couleurs autours ont comme luminosité.
            La méthode Malvar-Cutler devrait corriger cela en utilisant le gradient du canal vert 
            pour faire l'interpolation du rouge et du bleu. Les deux algorithmes sont linéaire et donc peu couteux,
            mais Malvar prend un temps un peu plus long. Cela est probablement dû au fait que la convolution se fait sur
            des matrices plus larges. Cependant, quand je révise mes résultats
            obtenus avec les algorithmes, la différence des artefacts semble négligeable, même en zoomant aux extrémités.
            Tout de même, les photos ont l'allure attendue par le rematriçage.
        """
        '</p>'
        '</div>'
    )
    
    content += section("Section 2: Dématriçage (Demosaicking)", sec2_content, icon="🎨")
    
    # =============================================================================
    # SECTION 3: Balance des Blancs (White Balance)
    # =============================================================================
    sec3_content = ""
    
    # Texte d'introduction pour la section 3
    sec3_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #e94560;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        "La balance des blancs sert à corriger la teinte globale de l'image pour "
        "que les objets neutres soit gris. L'algorithme Grey World dit que la moyenne de l'image "
        "est grise, ce qui est mauvais pour les images avec une couleur dominante (comme ma photo de forêt verte). "
        "La méthode de la région neutre automatique est meilleure si une zone blanche est présente. "
        '</p>'
        '</div>'
    )
    
    for basename in basenames:
        sec3_img_content = ""
        
        # Figure: Comparaison des méthodes
        comp_path = os.path.join(sec3_dir, f"{basename}_comparison.png")
        if os.path.exists(comp_path):
            sec3_img_content += subsection(
                f"Comparaison des méthodes - {basename}",
                figure(f"../images_intermediaires_sec3/{basename}_comparison.png",
                       "Comparaison des méthodes de balance des blancs")
            )
        
        # Figure: Conversion XYZ
        xyz_path = os.path.join(sec3_dir, f"{basename}_xyz_comparison.png")
        if os.path.exists(xyz_path):
            sec3_img_content += subsection(
                f"Conversion XYZ - {basename}",
                figure(f"../images_intermediaires_sec3/{basename}_xyz_comparison.png",
                       "Images converties en XYZ puis reconverties en sRGB")
            )
        
        if sec3_img_content:
            sec3_content += section(f"Image: {basename}", sec3_img_content)
    
    # Analyse et observations
    sec3_content += subsection(
        "Analyse et observations",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #e94560;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
    """
        Les multiplicateurs me dise la couleur dominante de la source lumineuse. 
        Par exemple, pour ma photo Ahsoka, cela signifie que le capteur a reçu beaucoup de rouge et qu'il faut le diminuer pour équilibrer l'image. 
        Ensuite, selon mes résultats, avec l'algorithme Malvar, l'algorithme de balance caméra obtient les meilleurs images. C'est l'algorithme qui utilise
        les raw datas directement, donc donne un meilleur point de vue du moment qui à été capturé. La photo Ahsoka à cependant des régions
        trop lumineuses où les lumières. Je crois que la balance caméra à de la misère avec les éclairage qui varie dans l'image.
        La conversion vers l'espace XYZ standardise les couleurs
        peut importe pour quel appareil, ce qui est nécessaire avant de convertir vers le sRGB.
    """
        '</p>'
        '</div>'
    )
    
    content += section("Section 3: Balance des Blancs (White Balance)", sec3_content, icon="⚪")
    
    # =============================================================================
    # SECTION 4: Mappage Tonal et Encodage d'Affichage
    # =============================================================================
    sec4_content = ""
    
    # Texte d'introduction pour la section 4
    sec4_content += subsection(
        "Introduction",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        "Le mappage tonal adapte la grande plage dynamique du capteur (comme HDR)"
        "à l'affichage limité. "
        "L'opérateur linéaire coupe les hautes lumières mais en "
        "perdant les détails dans les zones claires. L'opérateur de Reinhard compresse ces hautes "
        "lumières avec une courbe non-linéaire, préservant les détails. L'OETF sRGB applique "
        " une correction gamma pour que la luminosité "
        "perçue soit correcte. Finalament, la compression JPEG fait apparaître des artefacts de blocs visibles "
        "à cause de la perte d'information des hautes fréquences dans la compression."
        '</p>'
        '</div>'
    )
    
    # Concepts et algorithmes
    algorithms = algorithm_box(
        "A) Ajustement de luminosité",
        "<p>Division par le 99e percentile.</p>",
    )
    algorithms += algorithm_box(
        "B) Mappage tonal",
        "<p><b>Linéaire:</b> Pas de compression.</p>"
        "<p><b>Reinhard:</b> <code>L_out = L_in / (1 + L_in)</code>.</p>",
    )
    algorithms += algorithm_box(
        "C) Conversion XYZ → sRGB",
        "<p>Matrice standard D65 suivie de l'OETF sRGB.</p>",
    )
    algorithms += algorithm_box(
        "D) OETF sRGB",
        formula_box("sRGB = 1.055 × linéaire^(1/2.4) − 0.055")

    )
    algorithms += algorithm_box(
        "E) Analyse des artefacts JPEG",
        "<p>Sauvegarde en différentes qualités et analyse des artefacts.</p>",
    )
    
    sec4_content += subsection("Concepts et algorithmes", algorithms)
    
    # Figure: Courbes de mappage tonal
    curves_path = os.path.join(sec4_dir, "tonemapping_curves.png")
    if os.path.exists(curves_path):
        sec4_content += subsection(
            "Courbes de mappage tonal",
            figure("tonemapping_curves.png", "Comparaison des courbes de réponse")
        )
    
    # Figures pour chaque image
    # Utiliser results si disponible, sinon utiliser basenames
    # Filtrer pour ne garder que les 2 images sélectionnées
    if results:
        images_to_process = [r for r in results if r["basename"] in basenames]
    else:
        images_to_process = [{"basename": bn} for bn in basenames]
    
    for result in images_to_process:
        basename = result["basename"]
        dr = result.get("dynamic_range", {})
        
        sec4_img_content = ""
        
        # Figure: Comparaison des opérateurs
        comp_path = os.path.join(sec4_dir, f"{basename}_tonemapping_comparison.png")
        if os.path.exists(comp_path):
            sec4_img_content += subsection(
                "Comparaison des opérateurs",
                figure(
                    f"{basename}_tonemapping_comparison.png",
                    "Comparaison: Linéaire, Reinhard",
                ),
            )
        
        # Figure: Avant/Après OETF
        oetf_path = os.path.join(sec4_dir, f"{basename}_oetf_comparison.png")
        if os.path.exists(oetf_path):
            sec4_img_content += subsection(
                "Avant/Après OETF",
                figure(
                    f"{basename}_oetf_comparison.png",
                    "L'OETF encode les valeurs linéaires pour l'affichage",
                ),
            )
        
        # Figure: Image finale
        final_path = os.path.join(sec4_dir, f"{basename}_final.jpg")
        if os.path.exists(final_path):
            sec4_img_content += subsection(
                "Image finale",
                figure(f"{basename}_final.jpg", "Image JPEG finale (qualité 95)"),
            )
        
        # Figure: Plage dynamique
        dr_path = os.path.join(sec4_dir, f"{basename}_dynamic_range.png")
        if os.path.exists(dr_path):
            dr_table = ""
            if dr:
                dr_table = table(
                    ["Métrique", "Valeur"],
                    [
                        [
                            "Plage dynamique",
                            f"{dr.get('dynamic_range_stops', 0):.1f} stops",
                        ],
                        [
                            "Hautes lumières écrêtées",
                            f"{dr.get('highlight_clipped_percent', 0):.2f}%",
                        ],
                        ["Ombres écrasées", f"{dr.get('shadow_crushed_percent', 0):.2f}%"],
                    ],
                )
            sec4_img_content += subsection(
                "Plage dynamique",
                figure(
                    f"{basename}_dynamic_range.png", "Analyse des hautes lumières et ombres"
                ) + dr_table,
            )
            jpeg_metrics = result.get("jpeg_metrics", [])
            if jpeg_metrics:
                rows = []
                for m in jpeg_metrics:
                    rows.append([
                        f"JPEG Q={m['quality']}",
                        f"{m['size_kb']:.1f} KB",
                        f"{m['compression_ratio']:.1f}:1",
                        f"{m['psnr']:.2f} dB"
                    ])
                jpeg_table = table(
                    ["Qualité", "Taille", "Ratio", "PSNR"],
                    rows
                )
                sec4_img_content += subsection(
                    "Analyse de compression JPEG",
                    jpeg_table
                )

                artifacts_path = os.path.join(sec4_dir, f"{basename}_jpeg_artifacts.png")
                if os.path.exists(artifacts_path):
                    sec4_img_content += subsection(
                        "Artefacts visuels (Zoom)",
                        figure(f"{basename}_jpeg_artifacts.png", "Comparaison visuelle: Référence vs JPEG Q=25")
                    )
        
        if sec4_img_content:
            sec4_content += section(basename, sec4_img_content)
    
    # Analyse et observations
    sec4_content += subsection(
        "Analyse et observations", 
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #778da9;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        '''Comme discuté dans l'introduction, le mappage linéaire coupe les hautes lumières. On peut voir dans les photos les zones"
        "coupés. Les données nous dise 0.60% pour Ahsoka, ce qui est très faible donc peu de perte d'information."
        "Sans l'OETF sRGB, l'image linéaire est sombre et manque de lumière. L'OETF restaure la perception naturelle de la luminosité. L'image finale affiche une plage dynamique de 6.9 stops. 
        Les ombres sont plus écrasées (1.78%) que les hautes lumières. 
        Cela est probablement à cause du storage 8 bits qui ne peut pas contenir toute la plage dynamique capturée par le capteur (16 bits).
        Le tableau montre le compromis taille/qualité. 
        À Q=95, la qualité est maximale (PSNR 30.8 dB) mais le fichier est lourd (7.6 MB). 
        À Q=25, la compression est immense (63:1) et le fichier est de seulement 542 KB, mais la qualité visuelle n'est pas bonne (PSNR 27.35 dB et des artefacts de blocs sur les contours). 
        Je crois que Q=75 offre le meilleur compromis.'''
        '</p>'
        '</div>'
    )
    
    content += section("Section 4: Mappage Tonal et Encodage d'Affichage", sec4_content, icon="🎨")
    
    # =============================================================================
    # GRILLE DE COMPARAISON DES IMAGES FINALES
    # =============================================================================
    # Collecter toutes les images finales JPG de la section 4 et leurs références
    comparisons = []
    jpg_files = sorted(glob.glob(os.path.join(sec4_dir, "*_final.jpg")))
    
    for jpg_path in jpg_files:
        basename = os.path.basename(jpg_path).replace("_final.jpg", "")
        final_src = os.path.basename(jpg_path)
        
        # Chercher l'image de référence correspondante
        reference_src = None
        srgb_path = os.path.join(sec1_dir, f"{basename}_srgb.jpg")
        if os.path.exists(srgb_path):
            reference_src = f"../images_intermediaires_sec1/{basename}_srgb.jpg"
        
        if reference_src:
            comparisons.append({
                "basename": basename,
                "final_src": final_src,
                "reference_src": reference_src,
                "final_alt": f"Image finale - {basename}",
                "reference_alt": f"Référence sRGB - {basename}"
            })
        else:
            # Si pas de référence, ajouter quand même l'image finale seule
            comparisons.append({
                "basename": basename,
                "final_src": final_src,
                "reference_src": final_src,  # Dupliquer pour l'affichage
                "final_alt": f"Image finale - {basename}",
                "reference_alt": f"Image finale - {basename}"
            })
    
    if comparisons:
        grid_content = subsection(
            "Comparaison: Vos résultats vs Références sRGB",
            '<p style="color: #a0a0a0; margin-bottom: 20px;">Comparez vos images finales avec les aperçus sRGB générés par rawpy. Cliquez sur une image pour l\'agrandir.</p>'
        )
        grid_content += comparison_grid(comparisons)
        content += section("Comparaison des Images Finales", grid_content, icon="🖼️")
    
    # =============================================================================
    # CONCLUSION GÉNÉRALE
    # =============================================================================
    conclusion_content = subsection(
        "Conclusion",
        '<div style="background: rgba(0,0,0,0.2); padding: 20px; border-radius: 8px; margin: 20px 0; border-left: 4px solid #ffd54f;">'
        '<p style="color: #a0a0a0; font-style: italic;">'
        '''
        Le pipeline a permis de refaire la formation d'une image à partir des données raw venant d'une capteur.
        Dans mon cas, j'utilisais un Canon EOS REBEL T3i avec les réglage suivant : 
        - ISO-2500
        - F-stop f/2
        - Exposure time 1/160 sec
        - Bias 0 step
        - Metering mode Pattern
        En partant des données brutes du capteur, j'ai fait la normalisation, dématriçage, balance des blancs, et mappage tonal.
        J'ai constater l'importance du mappage tonal (Reinhard) pour compresser la dynamique 14 bits vers l'affichage 8 bits sans perdre les hautes lumières.
        La qualité est beaucoup moins bonne avec les pertes d'informations.
        Finalement, en comparant, mes images sorties du pipeline sont très semblables aux références, 
        quoique que les miennes semble trop clair.
        '''
        '</p>'
        '</div>'
    )
    
    content += section("Conclusion", conclusion_content, icon="📝")
    
    # Générer le document HTML final
    html = html_document(
        "Rapport TP1 - Maxime Bédard",
        "",
        "📸",
        content,
        accent_color="#778da9",
    )
    
    save_report(html, os.path.join(output_dir, "rapport_complet.html"))


# =============================================================================
# Traitement Principal
# =============================================================================


def process_display_encoding(
    input_dir="images_intermediaires_sec3",
    output_dir="images_intermediaires_sec4",
    input_suffix="_camera_xyz.tiff",
):
    """Traiter les images XYZ avec mappage tonal et encodage d'affichage."""
    os.makedirs(output_dir, exist_ok=True)

    tiff_files = sorted(glob.glob(os.path.join(input_dir, f"*{input_suffix}")))

    if not tiff_files:
        print(f"Aucun fichier *{input_suffix} trouvé dans {input_dir}/")
        return

    print(f"\n{'#'*60}")
    print("# Section 4: Mappage Tonal et Encodage d'Affichage")
    print(f"{'#'*60}")
    print(f"\n{len(tiff_files)} fichier(s) trouvé(s)")

    # Générer la figure des courbes une seule fois
    create_tonemapping_curves_figure(os.path.join(output_dir, "tonemapping_curves.png"))

    results = []

    for tiff_path in tiff_files:
        basename = os.path.basename(tiff_path).replace(input_suffix, "")

        print(f"\n{'='*60}")
        print(f"Traitement: {basename}")
        print("=" * 60)

        try:
            xyz_image = load_tiff(tiff_path)
            result = {"basename": basename}

            # Ajustement de luminosité (à implémenter par l'étudiant)
            print("  [0] Ajustement de luminosité...")
            xyz_image = adjust_brightness(xyz_image, percentile=99)

            # Comparaison des opérateurs de mappage tonal
            print("  [A] Comparaison du mappage tonal...")
            tonemap_funcs = {
                "Linéaire": tonemap_linear,
                "Reinhard": tonemap_reinhard,
            }
            srgb_results = create_tonemapping_comparison_figure(
                xyz_image,
                os.path.join(output_dir, f"{basename}_tonemapping_comparison.png"),
                tonemap_funcs,
                xyz_to_linear_srgb,
                linear_to_srgb,
                title=f"Mappage tonal - {basename}",
            )

            # Utiliser linéaire pour la suite (ou Reinhard si implémenté)
            xyz_tonemapped = tonemap_linear(xyz_image)
            rgb_linear = xyz_to_linear_srgb(xyz_tonemapped)
            rgb_linear = np.clip(rgb_linear, 0, 1)
            srgb = linear_to_srgb(rgb_linear)

            # Sauvegarder les résultats
            for name, img in srgb_results.items():
                save_tiff16(
                    img, os.path.join(output_dir, f"{basename}_{name.lower()}.tiff")
                )

            # Comparaison OETF
            print("  [B] Comparaison OETF...")
            create_oetf_comparison_figure(
                rgb_linear,
                srgb,
                os.path.join(output_dir, f"{basename}_oetf_comparison.png"),
                title=f"OETF sRGB - {basename}",
            )

            # Sauvegarder l'image finale en JPEG
            print("  [C] Sauvegarde de l'image finale...")
            img_8bit = quantize_to_8bit(srgb)

            final_jpg = os.path.join(output_dir, f"{basename}_final.jpg")
            save_jpeg(img_8bit, final_jpg, quality=95)

            # - Sauvegarder en différentes qualités (95, 75, 50, 25)
            # - Comparer avec PNG (sans perte)
            # - Visualiser les artefacts de compression
            # - Créer un graphique taille vs qualité
            print("  [!] Analyse des artefacts JPEG...")
            ref_png = os.path.join(output_dir, f"{basename}_ref.png") # png
            save_png(img_8bit, ref_png)

            ref_taille_kb = os.path.getsize(ref_png) / 1024.0 # la taille de l'image
            img_ref_float = img_8bit.astype(np.float32) / 255.0 #float

            jpeg_data = []
            qualites = [95, 75, 50, 25]
            for q in qualites:
                jpg_path = os.path.join(output_dir, f"{basename}_q{q}.jpg") # sauvegarde
                save_jpeg(img_8bit, jpg_path, quality=q)
                taille_kb = os.path.getsize(jpg_path) / 1024.0
                ratio = ref_taille_kb / taille_kb if taille_kb > 0 else 0

                img_jpg_loaded = np.array(Image.open(jpg_path)).astype(np.float32) / 255.0
                mse = np.mean((img_ref_float - img_jpg_loaded) ** 2)
                if mse == 0:
                    psnr = float('inf')
                else:
                    psnr = 10 * np.log10(1.0 / mse)
                jpeg_data.append({
                    "quality": q,
                    "size_kb": taille_kb,
                    "psnr": psnr,
                    "compression_ratio": ratio
                })
            result["jpeg_metrics"] = jpeg_data

            # Analyse de plage dynamique
            print("  [D] Analyse de plage dynamique...")
            dr_analysis = analyze_dynamic_range(rgb_linear)
            result["dynamic_range"] = dr_analysis
            print(
                f"    Plage dynamique: {dr_analysis['dynamic_range_stops']:.1f} stops"
            )

            create_dynamic_range_figure(
                rgb_linear,
                srgb,
                dr_analysis,
                os.path.join(output_dir, f"{basename}_dynamic_range.png"),
                title=f"Plage dynamique - {basename}",
            )

            results.append(result)

        except Exception as e:
            print(f"\nErreur lors du traitement de {tiff_path}: {e}")
            import traceback

            traceback.print_exc()

    if results:
        generate_report(results, output_dir)

    print(f"\n{'='*60}")
    print(f"Terminé! {len(results)} image(s) traitée(s) → {output_dir}/")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="TP1 Section 4: Mappage Tonal et Encodage"
    )
    parser.add_argument("--input-dir", "-i", default="images_intermediaires_sec3")
    parser.add_argument("--output-dir", "-o", default="images_intermediaires_sec4")
    parser.add_argument("--suffix", "-s", default="_camera_xyz.tiff")

    args = parser.parse_args()
    process_display_encoding(args.input_dir, args.output_dir, args.suffix)
