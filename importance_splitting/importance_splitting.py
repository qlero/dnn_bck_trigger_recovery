import math
import torch
import torchvision.transforms as tt
import os
import numpy as np

from torchvision.utils import save_image
from torchvision.io import read_image

from backdoor_injection import save_image

class ImportanceSplitting():
    def __init__(self, target, outer_iterations, inner_iterations, nb_candidates, strength, decay, name):
        self.target_class     = target
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

    def scorer(self, model, data, candidates, normalize, denormalize):
        pass

    def engine(self, model, data, normalize, denormalize, device):

        # INITIAL MONTE CARLO SAMPLING
        candidates  = self.generator(self.nb_candidates, data.shape, device)
        probabilities = []
        thresholds    = []

        data = data.to(device)
        
        with torch.no_grad():

            for out_it in range(self.outer_iterations):
                
                # Computes current scores and splits the candidates between survivors and those to be redrawn
                outputs     = self.scorer(model, data, candidates, normalize, denormalize)
                best_of_n   = torch.argsort(outputs, 0, descending=True)
                candidates  = candidates[best_of_n]
                outputs     = outputs[best_of_n]
                
                # Updates the cutoff threshold between survivors and to be redrawn
                self.threshold = outputs[self.cutoff].detach().cpu().item()
                print(f"[INFO] iteration {str(out_it+1).zfill(len(str(self.outer_iterations)))}/{self.outer_iterations}, threshold: {self.threshold:.3f}")
                probabilities.append(1/(2**(out_it+1)))
                thresholds.append(self.threshold)

                regenerated_candidates  = candidates[self.cutoff:]
                regenerated_outputs     = outputs[self.cutoff:]
                regenerated_placeholder = regenerated_candidates.clone().to(device)

                for in_it in range(self.inner_iterations):

                    # print(torch.cuda.mem_get_info())

                    # Regenerates and scores candidates
                    regenerated_candidates = self.kernel(regenerated_candidates, device)
                    outputs                = self.scorer(model, data, regenerated_candidates, normalize, denormalize)
                    # Finds which regenerations go above the threshold
                    mask = torch.logical_and(outputs > self.threshold, outputs > regenerated_outputs)
                    # Records the ones that are
                    regenerated_placeholder[mask] = regenerated_candidates[mask]
                    regenerated_outputs[mask]     = outputs[mask].to(device)
                    # Erases current run
                    regenerated_candidates = regenerated_placeholder.clone().to(device)

                # Updates strength ratio if needed
                # if torch.sum(mask)/len(mask) > 2/3:
                #     self.strength *= (1-self.decay)
                #     print(f"[INFO] updated strength down: {self.strength}")
                # if torch.sum(mask)/len(mask) < 1/3:
                #     self.strength /= (1-self.decay)
                #     self.strength = min(self.strength, 1)
                #     print(f"[INFO] updated strength up: {self.strength}")
                self.strength -= 1/self.outer_iterations

                # Updates the candidates' list
                candidates[self.cutoff:] = regenerated_candidates

                # Saves recovered triggers
                c = candidates[:self.cutoff].clone()
                c[c==torch.tensor(float("inf"))] = 0.5

                mean_candidate = torch.mean(c, axis=0, keepdim=True)
                save_image(self.format_to_image(mean_candidate[0]).detach().cpu(), f"example_recoveries/importance_splitting_{self.name}_target_{self.target_class}.png")

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

    def scorer(self, model, data, candidates, normalize, denormalize):
        # duplicates data and candidates along each other's axis and candidates number
        d = data.reshape(data.shape[0], 1, *data.shape[1:]).repeat(1, candidates.shape[0], *[1]*(len(data.shape)-1))
        c = candidates.reshape(1, *candidates.shape).repeat(data.shape[0], *[1]*len(data.shape))
        # Injects the candidates into the data and scores
        x      = normalize((denormalize(d) + (2*self.gaussian_cdf(c)-1)).clip(0,1))
        x      = x.reshape(data.shape[0]*candidates.shape[0], *data.shape[1:])
        scores = model(x)[:, self.target_class].reshape(data.shape[0], candidates.shape[0])
        scores = torch.mean(scores, axis=0)
        return scores


class IIDImportanceSplitting(ImportanceSplitting):

    def __init__(self, target, outer_iterations, inner_iterations, nb_candidates, strength, decay, sparseness, name):
        self.sparseness = sparseness
        super().__init__(target, outer_iterations, inner_iterations, nb_candidates, strength, decay, name)

    def format_to_image(self, candidates):
        c = candidates.clone()
        c[c==torch.tensor(float("inf"))] = 0.5
        c = c.repeat(3,1,1)
        return c

    def generator(self, draws, shape, device):
        mask = torch.rand(draws, 1, *shape[2:]) > self.sparseness
        data = torch.rand(draws, 1, *shape[2:]).to(device)
        data[mask] = torch.tensor(float("inf"))
        return data

    def kernel(self, candidates, device):
        mask = torch.rand(candidates.shape).to(device) < 0.5 #self.strength
        rep = self.generator(self.nb_to_redraw, candidates.shape, device)
        ret = torch.zeros(candidates.shape).to(device)
        ret[:,:,:,:] = torch.tensor(float("inf"))
        ret[torch.logical_not(mask)] = candidates[torch.logical_not(mask)]
        ret[torch.logical_and(mask, rep != torch.tensor(float("inf")))] = rep[torch.logical_and(mask, rep != torch.tensor(float("inf")))]
        ret[torch.logical_and(mask, rep == torch.tensor(float("inf")))] = ret[torch.logical_and(mask, rep == torch.tensor(float("inf")))]
        return ret

    def scorer(self, model, data, candidates, normalize, denormalize):
        # duplicates data and candidates along each other's axis and candidates number
        d = denormalize(data.reshape(data.shape[0], 1, *data.shape[1:]).repeat(1, candidates.shape[0], *[1]*(len(data.shape)-1)).clone())
        c = candidates.reshape(1, *candidates.shape).repeat(data.shape[0], 1, 3, *[1]*(len(data.shape)-2))
        # Injects the candidates into the data and scores
        m = c!=torch.tensor(float("inf"))
        d[m] = c[m]
        d    = d.reshape(data.shape[0]*candidates.shape[0], *data.shape[1:])
        scores = model(normalize(d))[:, self.target_class].reshape(data.shape[0], candidates.shape[0])
        scores = torch.mean(scores, axis=0)
        assert(list(scores.shape) == [candidates.shape[0]])
        return scores
