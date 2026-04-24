import sys
import os
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
import numpy as np
import torch
from typing import List, Tuple, Dict, Optional
import config
from model.network import HeartDiseaseNet, train_one_round, evaluate


class HospitalClient:
    def __init__(
        self,
        hospital_id:  str,
        X_train:      np.ndarray,
        y_train:      np.ndarray,
        X_test:       np.ndarray,
        y_test:       np.ndarray,
        proximal_mu:  float = config.PROXIMAL_MU,
    ):
        self.hospital_id  = hospital_id
        self.X_train      = X_train
        self.y_train      = y_train
        self.X_test       = X_test
        self.y_test       = y_test
        self.proximal_mu  = proximal_mu
        self.n_samples    = len(X_train)
        self.device       = torch.device("cpu")

        self.model = HeartDiseaseNet()


    def set_parameters(self, parameters: List[np.ndarray]):
        self.model.set_parameters(parameters)

    def get_parameters(self) -> List[np.ndarray]:

        return self.model.get_parameters()


    def local_fit(
        self,
        learning_rate:  float,
        local_epochs:   int,
        global_params:  Optional[List[np.ndarray]] = None,
    ) -> Tuple[List[np.ndarray], int, Dict]:

        frozen_global = None
        if self.proximal_mu > 0 and global_params is not None:
            param_keys = {name for name, _ in self.model.named_parameters()}
            state_keys = list(self.model.state_dict().keys())
            frozen_global = [
                torch.tensor(global_params[i].copy()).to(self.device)
                for i, k in enumerate(state_keys)
                if k in param_keys
            ]

        loss = train_one_round(
            model=self.model,
            X=self.X_train,
            y=self.y_train,
            epochs=local_epochs,
            lr=learning_rate,
            device=self.device,
            proximal_mu=self.proximal_mu,
            global_params=frozen_global,
        )

        return self.get_parameters(), self.n_samples, {
            "hospital":   self.hospital_id,
            "train_loss": float(loss),
        }


    def local_evaluate(self) -> Dict:
        """
        Évalue le modèle courant sur l'ensemble de test partagé.
        Retourne toutes les métriques de classification.
        """
        metrics = evaluate(self.model, self.X_test, self.y_test, self.device)
        return {"hospital": self.hospital_id, **metrics}