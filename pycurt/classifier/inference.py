import numpy as np
import torch
from pycurt.classifier.preprocessing import run_preprocessing
from pycurt.classifier.dataloader import BpClassDataLoader2D
from collections import Counter
import matplotlib.pyplot as plot


def extract_slices_inference(patient_data, directions):

    toinfer = []
    if 1 in patient_data.shape[1:]:
        indx = np.where(np.asarray(patient_data.shape[1:])==1)[0][0]
        directions = [indx+1]
    for direction in directions:
        if patient_data.shape[direction] > 1:
            slices = np.arange(np.ceil(np.percentile(np.arange(patient_data.shape[direction]), 25)), 
                               np.ceil(np.percentile(np.arange(patient_data.shape[direction]), 95)))
        else:
            slices = [0]
        for slice_idx in slices:
            slice_idx = int(slice_idx)
            if direction == 1:
                toinfer.append(patient_data[:, slice_idx, :, :])
            elif direction == 2:
                toinfer.append(patient_data[:, :, slice_idx, :])
            elif direction == 3:
                toinfer.append(patient_data[:, :, :, slice_idx])

    return toinfer


def load_checkpoint(filepath):
    
    if torch.cuda.is_available():
        map_location=lambda storage, loc: storage.cuda()
    else:
        map_location='cpu'
    checkpoint = torch.load(filepath, map_location=map_location)
    
    model = checkpoint['model']
    model.load_state_dict(checkpoint['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.eval()

    return model


def run_inference(for_inference, filepaths, modality='ct', body_parts=[], th=None):

    if modality == 'ct':
        directions = [1, 2, 3]
        if th is None:
            th = 0.33
    elif modality == 'mr':
        directions = [3]
        if th is None:
            th = 0.5

    patch_size = (400, 400)
    class_names = filepaths.keys()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model_dict = {}
    for key in filepaths:
        model_dict[key] = load_checkpoint(filepaths[key])

    result_dict = {}
    for el, image in enumerate(for_inference):
        result_dict[image] = {}
        print('Process # {}, image: {}'.format(el+1, image))
        try:
            preproc_images = run_preprocessing([image], modality)
        except:
            print('Pre-processing failed! Image will be ignored!')
            continue
        if preproc_images is None:
            print('Image has only one slice, it will be ignored!')
            continue
        slices_all = extract_slices_inference(preproc_images, directions)
        slice_labels = {}
        labs = []
        dataloader = BpClassDataLoader2D(slices_all, 1, patch_size, 1,
                                         shuffle=False, crop=False)

        for step, data in enumerate(dataloader):
            slice_labels[step] = {}
            inputs = data['data']
            inputs = torch.from_numpy(np.float32(inputs))
            inputs = inputs.to(device)
            for key in model_dict.keys():
                model = model_dict[key]
                output = model(inputs)
                prob = output.data.cpu().numpy()
                actRange = abs(prob[0][0])+abs(prob[0][1])
                index = output.data.cpu().numpy().argmax()
                if index == 1:
                    slice_labels[step][key] = prob[0, 1]
                else:
                    slice_labels[step]["other"] = prob[0, 0]

        val = {}
        for k in list(class_names):
            val[k] = []
            for slice_id in slice_labels.keys():
                labs = list(slice_labels[slice_id].keys())
                if k in labs:
                    try:
                        val[k].append(slice_labels[slice_id][k])
                    except:
                        print()
                else:
                    val[k].append(0)
            
        labels = []
        for slice_id in slice_labels.keys():
            labs = list(slice_labels[slice_id].keys())
            if "other" in labs:
                labs.remove("other")
            if labs and len(labs) == 1:
                labels.append(labs[0])
            elif labs and len(labs) > 1:
                probs = [slice_labels[slice_id][x] for x in labs]
                max_index = np.asarray(probs).argmax()
                labels.append(labs[max_index])
        if not labels:
            result_dict[image] = ["other", 0]
        else:
            c = Counter(labels)
            cn = list(c.keys())
            tot_predicted = sum([c[x] for x in c])
            if len(cn) > 1:
                temp = [[x, np.round(c[x]/tot_predicted, 2)] for x in cn
                         if np.round(c[x]/tot_predicted, 2) >= th]
                probs = [x[1] for x in temp]
                cn = [x[0] for x in temp]
                if cn:
                    bp_intersection = [[x, probs[i]] for i, x in enumerate(cn)
                                       if x in body_parts]
                    if bp_intersection and len(bp_intersection) == 1:
                        result_dict[image] = bp_intersection[0]
                    elif bp_intersection and len(bp_intersection) > 1:
                        probs = [x[1] for x in bp_intersection]
                        max_index = np.asarray(probs).argmax()
                        result_dict[image] = bp_intersection[max_index]
                    else:
                        result_dict[image] = ["other", 0]
                else:
                    result_dict[image] = ["other", 0]     
            else:
                result_dict[image] = [cn[0], 1]

        print('Image has been classified as "{}"  with probability of {}'.format(
            result_dict[image][0], result_dict[image][1]))

    return result_dict
