"""Comparación física con umbral de emparejamiento explícito, sin nueva inferencia."""
def comparar_distancias(matches, predicted, reference, threshold):
    if not 0 <= threshold <= 1:
        raise ValueError('IoU debe estar entre 0 y 1')
    pairs = {r['pred']: r['gt'] for r in matches
             if r['pred'] is not None and r['gt'] is not None and r['iou'] >= threshold}
    gt = {r['instance']: r for r in reference}
    errors = []
    for row in predicted:
        target = gt.get(pairs.get(row['instance']))
        if target is not None and pairs.get(row['principal']) == target['principal']:
            errors.append(abs(row['distance_surface_voxel_centers_mm'] - target['distance_surface_voxel_centers_mm']))
    return {'distance_min_iou': threshold, 'distance_valid_pairs': len(errors),
            'distance_MAE_mm': sum(errors) / len(errors) if errors else None}
