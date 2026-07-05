# ============================================================================
# IMPORTS DES MODÈLES ET MÉMOIRES
# ============================================================================
from src.core.memory import WorkingMemory, LearningMemory
from src.core.fusion import ModelFusion
from src.models.statistical import StatisticalModel
from src.models.bayesian import BayesianModel
from src.models.timeseries import TimeSeriesModel
from src.models.ml_model import MLModel
from src.models.anomaly_detection import AnomalyDetector
from src.models.deep_learning import DeepLearningModel  # ✅ NOUVEAU
from src.models.causal_model import CausalModel         # ✅ NOUVEAU
from src.models.optimization_model import OptimizationModel  # ✅ NOUVEAU

# ============================================================================
# INITIALISATION
# ============================================================================
# Mémoires
working_mem = WorkingMemory()
learning_mem = LearningMemory()

# Modèles
statistical_model = StatisticalModel()
bayesian_model = BayesianModel()
timeseries_model = TimeSeriesModel()
ml_model = MLModel()
anomaly_model = AnomalyDetector()
deep_learning_model = DeepLearningModel()  # ✅ NOUVEAU
causal_model = CausalModel()                 # ✅ NOUVEAU
optimization_model = OptimizationModel()   # ✅ NOUVEAU

# Fusion
fusion = ModelFusion(learning_mem)
fusion.models = {
    "statistical": statistical_model,
    "bayesian": bayesian_model,
    "timeseries": timeseries_model,
    "ml": ml_model,
    "anomaly": anomaly_model,
    "deep_learning": deep_learning_model,  # ✅ NOUVEAU
    "causal": causal_model,                 # ✅ NOUVEAU
    "optimization": optimization_model      # ✅ NOUVEAU
}

# Initialiser les poids par défaut
for name, weight in fusion.default_weights.items():
    learning_mem.update_weight(name, weight)
