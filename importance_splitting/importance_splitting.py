import math
import torch
import torchvision.transforms as tt
import os
import numpy as np

from torchvision.utils import save_image
from torchvision.io import read_image

from backdoor_injection import save_image

class ImportanceSplitting():
    def __init__(self, outer_iterations, inner_iterations, nb_candidates, strength, decay, name):
        self.outer_iterations = outer_iterations
        self.inner_iterations = inner_iterations
        self.strength         = strength
        self.decay            = decay
        self.nb_candidates    = nb_candidates
        self.threshold        = -float("inf")
        self.cutoff           = nb_candidates // 2
        self.nb_to_redraw     = nb_candidates - self.cutoff
        self.name             = name

    def format_to_image(self, candidates):
        pass

    def generator(self, draws, shape, device):
        pass

    def kernel(self, candidates, device):
        pass

    def scorer(self, model, data, candidates, target_class, normalize, denormalize):
        pass

    def engine(self, model, data, target_class, denormalize, normalize, device):

        # INITIAL MONTE CARLO SAMPLING
        candidates  = self.generator(self.nb_candidates, data.shape, device)
        probabilities = []
        thresholds    = []

        data = data.to(device)
        
        for out_it in range(self.outer_iterations):

            # Computes current scores and splits the candidates between survivors and those to be redrawn
            outputs     = self.scorer(model, data, candidates, target_class, normalize, denormalize)
            best_of_n   = torch.argsort(outputs, 0, descending=True)
            candidates  = candidates[best_of_n]
            outputs     = outputs[best_of_n]
            
            # Updates the cutoff threshold between survivors and to be redrawn
            self.threshold = outputs[self.cutoff].detach().cpu().item()
            print(f"[INFO] iteration {out_it+1}/{self.outer_iterations}, threshold: {self.threshold}")
            probabilities.append(1/(2**(out_it+1)))
            thresholds.append(self.threshold)

            regenerated_candidates  = candidates[self.cutoff:]
            regenerated_placeholder = regenerated_candidates.clone().to(device)

            for in_it in range(self.inner_iterations):
                # Regenerates and scores candidates
                regenerated_candidates = self.kernel(regenerated_candidates, device)
                outputs                = self.scorer(model, data, regenerated_candidates, target_class, normalize, denormalize)
                # Finds which regenerations go above the threshold
                mask = outputs > self.threshold
                # Records the ones that are
                regenerated_placeholder[mask] = regenerated_candidates[mask]
                # Erases current run
                regenerated_candidates = regenerated_placeholder.clone().to(device)

            # Updates strength ratio if needed
            if torch.sum(mask)/len(mask) > 2/3:
                self.strength *= (1-self.decay)
                print(f"[INFO] updated strength down: {self.strength}")
            if torch.sum(mask)/len(mask) < 1/3:
                self.strength /= (1-self.decay)
                # self.strength = min(self.strength, 1)
                print(f"[INFO] updated strength up: {self.strength}")

            # Updates the candidates' list
            candidates[self.cutoff:] = regenerated_candidates

            # Saves recovered triggers
            mean_candidate = torch.mean(candidates, axis=0, keepdim=True)
            save_image(self.format_to_image(mean_candidate[0]).detach().cpu(), f"example_recoveries/importance_splitting_{self.name}_iter_{out_it}.png")

        return candidates, probabilities, thresholds


class GaussianImportanceSplitting(ImportanceSplitting):

    def format_to_image(self, candidates):
        return self.gaussian_cdf(candidates)

    def gaussian_cdf(self, value):
        return 0.5 * (1 + torch.erf((value) * 1 / math.sqrt(2)))

    def generator(self, draws, shape, device):
        return torch.randn(draws, *shape[1:]).to(device)

    def kernel(self, candidates, device):
        return (candidates + self.strength * self.generator(self.nb_to_redraw, candidates.shape, device)) / (math.sqrt(1+self.strength**2))

    def scorer(self, model, data, candidates, target_class, normalize, denormalize):
        # duplicates data and candidates along each other's axis and candidates number
        d = data.reshape(data.shape[0], 1, *data.shape[1:]).repeat(1, candidates.shape[0], *[1]*(len(data.shape)-1))
        c = candidates.reshape(1, *candidates.shape).repeat(data.shape[0], *[1]*len(data.shape))
        # Injects the candidates into the data and scores
        x      = normalize((denormalize(d) + (2*self.gaussian_cdf(c)-1)).clip(0,1))
        x      = x.reshape(data.shape[0]*candidates.shape[0], *data.shape[1:])
        scores = model(x)[:,target_class].reshape(data.shape[0], candidates.shape[0])
        scores = torch.mean(scores, axis=0)
        assert(list(scores.shape) == [candidates.shape[0]])
        return scores


# def iid_kernel(data, s, device):
#     mask = torch.rand(data.shape).to(device) < s
#     rep = iid_generator(len(data), device)
#     ret = torch.zeros(data.shape).to(device)
#     ret[:,:,:,:] = torch.tensor(float("inf"))
#     ret[torch.logical_not(mask)] = data[torch.logical_not(mask)]
#     ret[torch.logical_and(mask, rep != torch.tensor(float("inf")))] = rep[torch.logical_and(mask, rep != torch.tensor(float("inf")))]
#     ret[torch.logical_and(mask, rep == torch.tensor(float("inf")))] = ret[torch.logical_and(mask, rep == torch.tensor(float("inf")))]
#     return ret

# def iid_generator(n, device, alpha=0.1):
#     mask = torch.rand((n, 1, 48, 48)).to(device) > alpha
#     data = torch.rand((n, 1, 48, 48)).to(device)
#     data[mask] = torch.tensor(float("inf"))
#     return data

# def iid_scorer(data, delta, model, target_class, normalize):
#     d = data.clone()
#     dd = delta.repeat(len(d),3,1,1)
#     # print(d.shape, dd.shape)
#     d[dd!=torch.tensor(float("inf"))] = dd[dd!=torch.tensor(float("inf"))]
#     outputs = model(normalize(d))
#     score = torch.mean(outputs[:,target_class]).detach().cpu().item()
#     return score, d[0]

# def iid_importance_splitting(model, target_class, normalize, device):

#     outer_iterations = 100
#     inner_iterations = 40
#     strength         = 0.5
#     nb_candidates    = 12*4
#     candidates       = iid_generator(nb_candidates, device)
#     threshold        = -float("inf")
#     cutoff           = nb_candidates//2

#     images = torch.zeros(10, 3, 48, 48)
#     imgs = os.listdir("check_gtsrb/")[10:]
#     for i in range(10):
#         images[i] = tt.Resize((48,48))(read_image(f"./check_gtsrb/{imgs[i]}").reshape(1,3,64,64))
#     images = images/255.
#     images = images.to(device)

#     nb_candidates_to_redraw = nb_candidates - cutoff

#     for i in range(outer_iterations):

#         # Initial score
#         outputs = torch.zeros(len(candidates))
#         for j in range(len(candidates)):
#             score, img = iid_scorer(images, candidates[j:j+1], model, target_class, normalize)
#             if j == 0:
#                 save_image(img, "is_ex.png")
#             outputs[j] = score
#         # print(outputs.shape)
#         # Sorts the results by best score towards the target class
#         best_of_n   = torch.argsort(outputs, 0, descending=True)
#         candidates  = candidates[best_of_n]
#         outputs     = outputs[best_of_n]

#         # Updates the threshold of selection
#         threshold   = outputs[cutoff].detach().cpu().item()
#         print(f"[INFO] Importance Splitting outer iteration {i} -- logit threshold {threshold:.4f}")

#         for j in range(nb_candidates_to_redraw):
            
#             for _ in range(inner_iterations):

#                 u = np.random.randint(0, cutoff)
#                 baseline = candidates[u:u+1].clone()
#                 prior_score = outputs[j]

#                 # Retrieves the current candidates
#                 mod = iid_kernel(baseline, strength, device)
#                 score, img = iid_scorer(images, mod, model, target_class, normalize)

#                 if score > threshold and score > prior_score:
#                     # candidates[j+cutoff:j+cutoff+1] = torch.mean(mod, 1, keepdim=True)
#                     candidates[j+cutoff:j+cutoff+1] = mod[:,0,:,:]
#                     prior_score = score

#             print(f"[INFO] Updated element {j} with score {score} vs threshold {threshold}")

#             assert(len(candidates) == nb_candidates)
        
#         result = candidates[:cutoff].clone()
#         result[result==torch.tensor(float("inf"))]=0.5
#         save_image(torch.mean(result,0), f"is_ex_it.png")

#     return result, images