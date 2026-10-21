
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# path vari
DATASET_DIR = Path("data/hands")
METADATA_DIR = Path("data/metadata")
MODEL_PATH = Path("models/hand_landmarker.task")

OUTPUT_DIR = Path("output")
ROI_DIR = OUTPUT_DIR / "rois"
OVERLAY_DIR = OUTPUT_DIR / "overlays"

ROI_DIR.mkdir(parents=True, exist_ok=True)
OVERLAY_DIR.mkdir(parents=True, exist_ok=True)

# coppie di landmark (base dito, articolazione intermedia)
PIP_JOINTS = {
    "index_pip": (5, 6),    # indice
    "middle_pip": (9, 10),  # medio
    "ring_pip": (13, 14),   # anulare
}

# collegamenti tra landmark per disegnare scheletro mano
CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17)
]


def normalize(vector):
    """
    Restituisce un vettore unitario; rifiuta vettori di lunghezza quasi nulla.
    """
    length = np.linalg.norm(vector)
    if length < 1e-6:
        raise ValueError("Landmark troppo vicini")
    return vector / length


def get_roi(image, center, x_axis, y_axis, width, height, output_size=224):
    """
    Ritaglia una ROI orientata e la normalizza a output_size x output_size.
    Center e gli assi definiscono il rettangolo sull’immagine originale.
    Restituisce il ritaglio e i quattro vertici in pixel per i controlli.
    """
    x_axis = normalize(x_axis)
    y_axis = normalize(y_axis)

    # semi-assi che permettono di costruire i quattro angoli del rettangolo
    half_x = x_axis * width / 2
    half_y = y_axis * height / 2

    corners = np.float32([
        center - half_x - half_y,
        center + half_x - half_y,
        center + half_x + half_y,
        center - half_x + half_y
    ])

    destination = np.float32([
        [0, 0],
        [output_size - 1, 0],
        [output_size - 1, output_size - 1],
        [0, output_size - 1]
    ])

    # trasforma il quadrilatero originale in un quadrato di dim. fissa
    matrix = cv2.getPerspectiveTransform(corners, destination)

    roi = cv2.warpPerspective(
        image,
        matrix,
        (output_size, output_size),
        borderMode=cv2.BORDER_CONSTANT,  # il bianco riempie eventuali zone esterne 
        borderValue=(255, 255, 255)
    )

    return roi, corners


def extract_palm_roi(image, points):
    """
    Definisce la ROI del palmo usando landmark vicini alle dita.
    Evita di dipendere dal landmark del polso, spesso fuori inquadratura.
    """
    index_mcp = points[5]
    middle_mcp = points[9]
    pinky_mcp = points[17]
    middle_pip = points[10]

    # punto medio tra le basi di indice e mignolo
    base_center = (index_mcp + pinky_mcp) / 2

    x_axis = normalize(pinky_mcp - index_mcp)

    finger_direction = normalize(middle_pip - middle_mcp)

    y_axis = np.array([-x_axis[1], x_axis[0]])

    # orienta l'asse longitudinale verso il polso
    if np.dot(y_axis, finger_direction) > 0:
        y_axis = -y_axis

    palm_width = np.linalg.norm(pinky_mcp - index_mcp)

    # coefficienti empirici scelti in base ai test (posizione e ampiezza ROI)
    # proporzionali alla larghezza del palmo
    center = base_center + y_axis * palm_width * 0.45

    return get_roi(
        image,
        center,
        x_axis,
        y_axis,
        width=palm_width * 0.75,
        height=palm_width * 0.65
    )


def extract_pip_roi(image, points, mcp_id, pip_id):
    """
    Estrae una ROI centrata sull’articolazione PIP di un dito dorsale.
    """
    mcp = points[mcp_id]
    pip = points[pip_id]

    # la direzione MCP -> PIP da l'orientamento del dito
    y_axis = normalize(pip - mcp)
    x_axis = np.array([-y_axis[1], y_axis[0]])

    segment_length = np.linalg.norm(pip - mcp)

    return get_roi(
        image,
        pip,
        x_axis,
        y_axis,
        width=segment_length * 0.55,
        height=segment_length * 0.65
    )


def draw_landmarks(image, points):
    """
    Disegna collegamenti, punti e indici dei 21 landmark sull’overlay.
    """
    for start, end in CONNECTIONS:
        a = tuple(np.round(points[start]).astype(int))
        b = tuple(np.round(points[end]).astype(int))
        cv2.line(image, a, b, (255, 180, 0), 2)

    for index, point in enumerate(points):
        x, y = np.round(point).astype(int)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
        cv2.putText(
            image,
            str(index),
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 0, 200),
            1
        )


def roi_inside_image(corners, image):
    """
    Verifica che tutti i vertici della ROI siano dentro l'immagine.
    """
    height, width = image.shape[:2]

    return np.all(
        (corners[:, 0] >= 0) &
        (corners[:, 0] < width) &
        (corners[:, 1] >= 0) &
        (corners[:, 1] < height)
    )



def detect_with_rotations(image, landmarker):
    """
    Cerca una mano a diverse risoluzioni e rotazioni.
    Al primo successo riconverte i landmark in pixel dell’immagine originale.
    Restituisce (punti, lato massimo richiesto, angolo), oppure tre None.
    """
    height, width = image.shape[:2]

    sizes = [640, 960, 1280]

    rotations = [
        (0, None),
        (90, cv2.ROTATE_90_CLOCKWISE),
        (180, cv2.ROTATE_180),
        (270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    for max_size in sizes:

        # mantiene l'aspect ratio (rapporto tra larghezza e altezza) e non ingrandisce immagini piccole 
        scale = min(1.0, max_size / max(height, width))

        resized = cv2.resize(
            image,
            None,
            fx=scale,
            fy=scale,
            interpolation=cv2.INTER_AREA
        )

        for angle, rotation in rotations:
            if rotation is None:
                rotated = resized
            else:
                rotated = cv2.rotate(resized, rotation)

            rgb = cv2.cvtColor(rotated, cv2.COLOR_BGR2RGB)

            mp_image = mp.Image(
                image_format=mp.ImageFormat.SRGB,
                data=np.ascontiguousarray(rgb)
            )

            result = landmarker.detect(mp_image)

            if not result.hand_landmarks:
                print(
                    f"Dimensione {max_size}, rotazione {angle}: "
                    "nessuna mano rilevata"
                )
                continue

            rotated_h, rotated_w = rotated.shape[:2]

            points = np.array([
                [p.x * rotated_w, p.y * rotated_h]
                for p in result.hand_landmarks[0]
            ], dtype=np.float32)

            resized_h, resized_w = resized.shape[:2]

            if angle == 90:
                points = np.column_stack([
                    points[:, 1],
                    resized_h - 1 - points[:, 0]
                ])

            elif angle == 180:
                points = np.column_stack([
                    resized_w - 1 - points[:, 0],
                    resized_h - 1 - points[:, 1]
                ])

            elif angle == 270:
                points = np.column_stack([
                    resized_w - 1 - points[:, 1],
                    points[:, 0]
                ])

            # fa tornare l'immagine alle coord. originali
            points[:, 0] *= width / resized_w
            points[:, 1] *= height / resized_h

            print(
                f"Mano rilevata con dimensione {max_size} "
                f"e rotazione {angle}"
            )

            return points.astype(np.float32), max_size, angle

    return None, None, None

def extract_geometry_features(points):
    """Calcola 16 descrittori geometrici 2D da 21 landmark palmari.
    Le distanze sono normalizzate rispetto alla distanza fra gli MCP
    di indice (5) e mignolo (17), per attenuare l'effetto della scala.
    Le misure dipendono comunque da posa e prospettiva della mano.
    """
    def dist(a, b):
        return float(np.linalg.norm(points[a] - points[b]))

    reference = dist(5, 17)
    if not np.isfinite(reference) or reference < 1e-6:
        raise ValueError("Riferimento geometrico non valido")

    # Lunghezza proiettata di ciascun dito: somma dei segmenti ossei.
    chains = {
        "thumb": [1, 2, 3, 4],
        "index": [5, 6, 7, 8],
        "middle": [9, 10, 11, 12],
        "ring": [13, 14, 15, 16],
        "pinky": [17, 18, 19, 20],
    }
    lengths = {
        finger: sum(dist(a, b) for a, b in zip(ids[:-1], ids[1:]))
        for finger, ids in chains.items()
    }

    features = {
        f"length_{finger}_norm": length / reference
        for finger, length in lengths.items()
    }

    # Distanze normalizzate fra basi (MCP) delle dita adiacenti.
    for name, a, b in [
        ("index_middle", 5, 9),
        ("middle_ring", 9, 13),
        ("ring_pinky", 13, 17),
    ]:
        features[f"mcp_{name}_norm"] = dist(a, b) / reference

    # Rapporti rispetto al dito medio, per confrontare proporzioni.
    middle_length = lengths["middle"]
    if middle_length < 1e-6:
        raise ValueError("Lunghezza dito medio non valida")
    for finger in ("thumb", "index", "ring", "pinky"):
        features[f"ratio_{finger}_middle"] = lengths[finger] / middle_length

    # Angoli di apertura tra dita vicine (sensibili alla posa).
    directions = {
        finger: points[ids[-1]] - points[ids[0]]
        for finger, ids in chains.items()
    }
    for first, second in [
        ("thumb", "index"),
        ("index", "middle"),
        ("middle", "ring"),
        ("ring", "pinky"),
    ]:
        a = normalize(directions[first])
        b = normalize(directions[second])
        cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
        features[f"angle_{first}_{second}_deg"] = float(
            np.degrees(np.arccos(cosine))
        )

    if not np.all(np.isfinite(list(features.values()))):
        raise ValueError("Feature geometriche non finite")
    return features


def process_image(image_path, aspect, landmarker):
    """
    Elabora un'immagine, salva ROI e overlay e ne restituisce le statistiche.
    Per il palmo ci si aspetta 1 ROI, per il dorso 3. 
    """
    image = cv2.imread(str(image_path))

    expected = 1 if aspect.startswith("palmar") else 3

    # statistiche per report in csv
    stats = {
        "hand_detected": False,
        "roi_expected": expected,
        "roi_saved": 0,
        "roi_discarded": 0,
        "max_size": None,
        "rotation": None,
        "status": "unknown",
        "geometry_status": "not_attempted"
    }

    if image is None:
        print(f"Impossibile aprire: {image_path}")
        stats["status"] = "image_error"
        return stats

    points, max_size, angle = detect_with_rotations(
        image, landmarker
    )

    if points is None:
        print(f"Nessuna mano rilevata: {image_path.name}")
        stats["status"] = "no_hand"
        return stats

    stats["hand_detected"] = True
    stats["max_size"] = max_size
    stats["rotation"] = angle

    try:
        stats["geometry_features"] = extract_geometry_features(points)
        stats["geometry_status"] = "success"
    except ValueError as error:
        stats["geometry_status"] = "invalid_geometry"
        print(f"Geometria non valida: {image_path.name} - {error}")

    overlay = image.copy()
    draw_landmarks(overlay, points)

    # si associa a ciascuna regione il ritaglio e i suoi vertici
    roi_results = {}

    if aspect.startswith("palmar"):
        roi_results["palm_center"] = extract_palm_roi(
            image, points
        )

    elif aspect.startswith("dorsal"):
        for name, (mcp_id, pip_id) in PIP_JOINTS.items():
            roi_results[name] = extract_pip_roi(
                image, points, mcp_id, pip_id
            )

    aspect_name = aspect.replace(" ", "_")
    prefix = f"{image_path.stem}_{aspect_name}"

    for name, (roi, corners) in roi_results.items():
        polygon = np.round(corners).astype(np.int32)

        # si scartano i ritagli con padding fuori dall'immagine originale
        if not roi_inside_image(corners, image):
            stats["roi_discarded"] += 1

            print(
                f"ROI fuori dai confini: "
                f"{image_path.name} - {name}"
            )

            cv2.polylines(
                overlay, [polygon], True, (0, 0, 255), 3
            )
            continue

        roi_path = ROI_DIR / f"{prefix}_{name}.jpg"

        saved = cv2.imwrite(str(roi_path), roi)

        if not saved:
            stats["roi_discarded"] += 1
            print(f"Errore salvataggio ROI: {roi_path}")
            continue

        stats["roi_saved"] += 1

        cv2.polylines(
            overlay, [polygon], True, (0, 255, 0), 3
        )

        label_position = tuple(polygon[0])

        cv2.putText(
            overlay,
            name,
            label_position,
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        print(f"ROI salvata: {roi_path}")

    overlay_path = OVERLAY_DIR / f"{prefix}_overlay.jpg"
    cv2.imwrite(str(overlay_path), overlay)

    print(f"Overlay salvato: {overlay_path}")

    if stats["roi_saved"] == stats["roi_expected"]:
        stats["status"] = "success"
    elif stats["roi_saved"] > 0:
        stats["status"] = "partial"
    else:
        stats["status"] = "no_valid_roi"

    return stats


def main():
    """
    Elabora 100 palmi e 100 dorsi, estraendo ROI e geometria da entrambi.
    """
    csv_files = sorted(METADATA_DIR.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError("CSV non trovato")

    df = pd.read_csv(csv_files[0])

    image_files = {
        path.name: path
        for path in DATASET_DIR.rglob("*.jpg")
    }

    sample_path = OUTPUT_DIR / "campione_geometria.csv"
    sample_df = pd.read_csv(sample_path)

    selected = []

    for _, row in sample_df.iterrows():
        image_path = image_files.get(row["imageName"])

        if image_path is None:
            print(f"Immagine non trovata: {row['imageName']}")
            continue

        selected.append((image_path, row))

    # soglie mediapipe predefinite
    options = vision.HandLandmarkerOptions(
        base_options=python.BaseOptions(
            model_asset_path=str(MODEL_PATH)
        ),
        running_mode=vision.RunningMode.IMAGE,
        num_hands=1,
        min_hand_detection_confidence=0.5,
        min_hand_presence_confidence=0.5
    )

    results = []
    geometry_results = []

    with vision.HandLandmarker.create_from_options(
        options
    ) as landmarker:

        for index, (image_path, row) in enumerate(selected, 1):
            print(
                f"\n[{index}/{len(selected)}] "
                f"Elaborazione: {image_path.name}"
            )

            try:
                stats = process_image(
                    image_path,
                    row["aspectOfHand"],
                    landmarker
                )

            except (ValueError, cv2.error) as error:
                print(f"Errore: {error}")

                stats = {
                    "hand_detected": False,
                    "roi_expected": (
                        1 if row["aspectOfHand"].startswith("palmar")
                        else 3
                    ),
                    "roi_saved": 0,
                    "roi_discarded": 0,
                    "max_size": None,
                    "rotation": None,
                    "status": "processing_error",
                    "geometry_status": "not_attempted"
                }

            geometry = stats.pop("geometry_features", None)
            if geometry is not None:
                geometry_results.append({
                    "imageName": row["imageName"],
                    "id": row["id"],
                    "aspectOfHand": row["aspectOfHand"],
                    **geometry
                })

            results.append({
                "imageName": row["imageName"],
                "id": row["id"],
                "aspectOfHand": row["aspectOfHand"],
                **stats
            })

    # esporta statistiche utili per il preprocessing
    results_df = pd.DataFrame(results)

    report_path = OUTPUT_DIR / "preprocessing_report_confronto.csv"
    results_df.to_csv(report_path, index=False)

    geometry_path = OUTPUT_DIR / "geometry_features_confronto.csv"
    pd.DataFrame(geometry_results).to_csv(geometry_path, index=False)

    print("\n===== RISULTATI FINALI =====")

    print(f"Immagini analizzate: {len(results_df)}")

    print(
        f"Mani rilevate: "
        f"{results_df['hand_detected'].sum()}"
    )

    print(
        f"ROI salvate: "
        f"{results_df['roi_saved'].sum()}"
    )

    print(
        f"ROI scartate: "
        f"{results_df['roi_discarded'].sum()}"
    )

    print("\nStato elaborazioni:")
    print(results_df["status"].value_counts())

    print(f"\nReport salvato in: {report_path}")
    
    geometry_df = pd.DataFrame(geometry_results)

    print("\n===== ESTRAZIONE GEOMETRICA =====")

    print(f"Immagini con geometria valida: {len(geometry_df)}")

    if not geometry_df.empty:
        print("\nPer superficie:")
        print(
            geometry_df["aspectOfHand"]
            .str.split()
            .str[0]
            .value_counts()
        )

    print(f"Geometria estratta da: {len(geometry_results)} immagini")

if __name__ == "__main__":
    main()