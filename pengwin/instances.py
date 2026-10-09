"""Instancias volumétricas y distancia EDT: solo volúmenes contiguos, nunca cortes salteados."""
import numpy as np
from scipy import ndimage as ndi
from scipy.optimize import linear_sum_assignment
from skimage.segmentation import watershed

def semantic_labels(ids):
    return np.where(ids>0,(ids.astype(np.int16)-1)//10+1,0).astype(np.uint8)

def fragment_boundaries(ids):
    """Interfaz de IDs distintos del mismo hueso, invariante a permutar sus números."""
    region=semantic_labels(ids); edge=np.zeros(ids.shape,bool)
    for axis in range(ids.ndim):
        a=[slice(None)]*ids.ndim;b=a.copy();a[axis]=slice(None,-1);b[axis]=slice(1,None)
        a,b=tuple(a),tuple(b)
        change=(ids[a]!=ids[b])&(ids[a]>0)&(ids[b]>0)&(region[a]==region[b])
        edge[a]|=change;edge[b]|=change
    return edge.astype(np.float32)

def reconstruct_instances(semantic,boundary,spacing_zyx,threshold=.5,min_volume_mm3=20.,seed_min_volume_mm3=0.):
    if semantic.ndim!=3 or semantic.shape!=boundary.shape: raise ValueError('Se requiere volumen 3D contiguo')
    spacing=np.asarray(spacing_zyx,float)
    if spacing.shape!=(3,) or not np.isfinite(spacing).all() or (spacing<=0).any(): raise ValueError('Spacing inválido')
    result=np.zeros(semantic.shape,np.int32); mapping={}; next_id=1
    structure=np.ones((3,3,3),bool)
    for region in (1,2,3):
        mask=semantic==region
        if not mask.any(): continue
        markers,n=ndi.label(mask&(boundary<threshold),structure)
        if seed_min_volume_mm3 > 0:
            counts=np.bincount(markers.ravel())
            keep=counts*np.prod(spacing)>=seed_min_volume_mm3
            keep[0]=False
            markers=markers*keep[markers]
        components,nc=ndi.label(mask,structure)
        # Todo componente sin semilla recibe una; no desaparecer por incertidumbre de borde.
        seeded=np.unique(components[markers>0])
        unseeded=np.setdiff1d(np.arange(1,nc+1),seeded)
        if len(unseeded):
            locations=ndi.minimum_position(boundary,labels=components,index=unseeded)
            for point in locations:
                n+=1;markers[point]=n
        assigned=watershed(boundary.astype(np.float32),markers,mask=mask,connectivity=structure)
        labels,counts=np.unique(assigned[assigned>0],return_counts=True)
        lookup=np.zeros(int(assigned.max())+1,np.int32)
        for label,count in zip(labels,counts):
            if count*np.prod(spacing)<min_volume_mm3: continue
            lookup[label]=next_id;mapping[next_id]=region;next_id+=1
        result[mask]=lookup[assigned[mask]]
    return result,mapping

def separation_distances(instances,mapping,spacing_zyx,principal_ids=None):
    """Mínimo entre centros de vóxeles de superficie (aproximación EDT al borde).

    No restar arbitrariamente un spacing a la diagonal. Contacto se informa aparte.
    """
    if instances.ndim!=3: raise ValueError('Distancias requieren 3D')
    sp=np.asarray(spacing_zyx,float)
    if sp.shape!=(3,) or not np.isfinite(sp).all() or (sp<=0).any(): raise ValueError('Spacing inválido')
    rows=[]
    volumes=np.bincount(instances.ravel())
    for region in (1,2,3):
        ids=[i for i,r in mapping.items() if r==region and i<len(volumes) and volumes[i]>0]
        if not ids: continue
        principal=max(ids,key=lambda i:volumes[i]) if principal_ids is None else principal_ids[region]
        if principal not in ids: raise ValueError(f'Principal GT ausente en región {region}')
        main=instances==principal
        surface=main&~ndi.binary_erosion(main,structure=ndi.generate_binary_structure(3,1))
        dist=ndi.distance_transform_edt(~surface,sampling=sp)
        adjacent=ndi.binary_dilation(main,structure=np.ones((3,3,3)))&~main
        minimums=ndi.minimum(dist,labels=instances,index=ids)
        contacts=ndi.maximum(adjacent,labels=instances,index=ids)
        for i,minimum,contact in zip(ids,minimums,contacts):
            if i==principal: continue
            rows.append({'instance':int(i),'region':region,'principal':int(principal),
                         'distance_surface_voxel_centers_mm':float(minimum),
                         'contact_26':bool(contact),'volume_mm3':float(volumes[i]*np.prod(sp))})
    return rows

def match_instances(pred,gt,pred_mapping):
    """Asignación 1:1 máxima IoU dentro de cada anatomía; omisiones y extras explícitos."""
    rows=[];present=set(np.unique(pred));gt_present=np.unique(gt)
    for r in (1,2,3):
        pp=[i for i,c in pred_mapping.items() if c==r and i in present]
        gg=[int(i) for i in gt_present if i>0 and (int(i)-1)//10+1==r]
        # Tabla de contingencia evita comparar un volumen completo por cada pareja.
        # Enmascarar otras regiones antes de indexar.
        plook=np.zeros(int(pred.max())+1,int)
        for j,i in enumerate(pp,1): plook[i]=j
        glook=np.zeros(max(31,int(gt.max())+1),int)
        for j,i in enumerate(gg,1): glook[i]=j
        table=np.bincount((plook[pred]*(len(gg)+1)+glook[gt]).ravel(),minlength=(len(pp)+1)*(len(gg)+1)).reshape(len(pp)+1,len(gg)+1)
        inter=table[1:,1:];ps=table[1:].sum(1);gs=table[:,1:].sum(0)
        union=ps[:,None]+gs[None,:]-inter
        iou=inter/np.maximum(union,1)
        usedp=set();usedg=set()
        if len(pp) and len(gg):
            ii,jj=linear_sum_assignment(-iou)
            for i,j in zip(ii,jj):
                if iou[i,j]<=0: continue
                usedp.add(i);usedg.add(j)
                rows.append({'region':r,'pred':pp[i],'gt':gg[j],'iou':float(iou[i,j]),'dice':float(2*inter[i,j]/(ps[i]+gs[j]))})
        rows.extend({'region':r,'pred':None,'gt':g,'iou':0.,'dice':0.} for j,g in enumerate(gg) if j not in usedg)
        rows.extend({'region':r,'pred':p,'gt':None,'iou':0.,'dice':0.} for i,p in enumerate(pp) if i not in usedp)
    return rows
