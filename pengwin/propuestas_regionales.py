"""Combinar propuestas propias: RPN primero, grid previo como respaldo."""
import numpy as np

def combinar_regiones(old_boxes,old_scores,old_labels,new_boxes,new_scores,new_labels,*,new_conf=.1,old_conf=.25,padding=.01):
    old_boxes=np.asarray(old_boxes).reshape(-1,4);new_boxes=np.asarray(new_boxes).reshape(-1,4)
    old_scores=np.asarray(old_scores);new_scores=np.asarray(new_scores);old_labels=np.asarray(old_labels);new_labels=np.asarray(new_labels)
    boxes=[];scores=[];labels=[];sources=[]
    for cls in range(3):
        ni=np.where((new_labels==cls)&(new_scores>=new_conf))[0]
        oi=np.where((old_labels==cls)&(old_scores>=old_conf))[0]
        if len(ni):j=ni[np.argmax(new_scores[ni])];box=new_boxes[j];score=new_scores[j];source='rpn_grid'
        elif len(oi):j=oi[np.argmax(old_scores[oi])];box=old_boxes[j];score=old_scores[j];source='grid_respaldo'
        else:continue
        boxes.append(np.clip(box+np.array([-padding,-padding,padding,padding]),0,1));scores.append(float(score));labels.append(cls);sources.append(source)
    return np.asarray(boxes).reshape(-1,4),np.asarray(scores),np.asarray(labels,dtype=np.int64),sources
