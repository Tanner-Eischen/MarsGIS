"""
Machine Learning Site Recommendation Engine for Mars Habitat Selection

This module implements advanced ML algorithms for intelligent site recommendation
based on terrain characteristics, resource availability, and mission constraints.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import calinski_harabasz_score, silhouette_score
from sklearn.preprocessing import StandardScaler


@dataclass
class SiteFeatures:
    """Comprehensive site feature set for ML analysis"""
    elevation: float
    slope: float
    aspect: float
    roughness: float
    tri: float  # Terrain Ruggedness Index
    solar_exposure: float
    proximity_to_water_ice: float
    proximity_to_minerals: float
    accessibility_score: float
    safety_score: float
    resource_potential: float
    environmental_stability: float

@dataclass
class SiteRecommendation:
    """ML-generated site recommendation with confidence metrics"""
    site_id: str
    coordinates: tuple[float, float]  # (lat, lon)
    overall_score: float
    feature_scores: dict[str, float]
    confidence: float
    recommendation_reasons: list[str]
    risk_factors: list[str]
    suitability_rank: int
    cluster_assignment: int

class MLSiteRecommendationEngine:
    """
    Advanced ML-powered site recommendation system for Mars habitat selection

    Features:
    - Multi-criteria optimization using ensemble methods
    - Unsupervised clustering for site categorization
    - Confidence-based ranking with uncertainty quantification
    - Adaptive learning from mission outcomes
    - Real-time recommendation updates
    """

    def __init__(self, model_dir: str = "models/ml_site_rec"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)

        # Feature weights for different mission types
        self.mission_weights = {
            'research_base': {
                'safety': 0.25, 'accessibility': 0.20, 'resources': 0.20,
                'environmental_stability': 0.15, 'solar_exposure': 0.10, 'terrain': 0.10
            },
            'mining_operation': {
                'resources': 0.30, 'accessibility': 0.25, 'safety': 0.20,
                'environmental_stability': 0.10, 'terrain': 0.10, 'solar_exposure': 0.05
            },
            'emergency_shelter': {
                'safety': 0.35, 'accessibility': 0.30, 'environmental_stability': 0.20,
                'terrain': 0.10, 'solar_exposure': 0.03, 'resources': 0.02
            },
            'permanent_settlement': {
                'environmental_stability': 0.25, 'safety': 0.20, 'resources': 0.20,
                'solar_exposure': 0.15, 'accessibility': 0.15, 'terrain': 0.05
            }
        }

        # ML Models
        self.ensemble_model = None
        self.clustering_model = None
        self.scaler = None
        self.feature_selector = None

        # Training data
        self.training_data = None
        self.site_clusters = None

        # Model performance metrics
        self.model_metrics = {}
        self.is_trained = False

    def extract_features_from_terrain_data(self, terrain_data: dict) -> SiteFeatures:
        """
        Extract comprehensive features from terrain analysis data

        Args:
            terrain_data: Dictionary containing terrain analysis results

        Returns:
            SiteFeatures object with extracted features
        """
        # Basic terrain metrics
        elevation = float(terrain_data.get('elevation', 0.0))
        slope = float(terrain_data.get('slope_mean', 0.0))
        aspect = float(terrain_data.get('aspect_mean', 0.0))
        roughness = float(terrain_data.get('roughness_mean', 0.0))
        tri = float(terrain_data.get('tri_mean', 0.0))

        # Calculate solar exposure based on aspect and slope
        solar_exposure = self._calculate_solar_exposure(slope, aspect)

        # Resource proximity (simulated based on known Mars resource maps)
        proximity_to_water_ice = self._calculate_resource_proximity(
            terrain_data.get('coordinates', (0, 0)), 'water_ice'
        )
        proximity_to_minerals = self._calculate_resource_proximity(
            terrain_data.get('coordinates', (0, 0)), 'minerals'
        )

        # Accessibility and safety scores
        accessibility_score = self._calculate_accessibility_score(terrain_data)
        safety_score = self._calculate_safety_score(terrain_data)

        # Resource potential and environmental stability
        resource_potential = self._calculate_resource_potential(
            proximity_to_water_ice, proximity_to_minerals, terrain_data
        )
        environmental_stability = self._calculate_environmental_stability(terrain_data)

        return SiteFeatures(
            elevation=elevation,
            slope=slope,
            aspect=aspect,
            roughness=roughness,
            tri=tri,
            solar_exposure=solar_exposure,
            proximity_to_water_ice=proximity_to_water_ice,
            proximity_to_minerals=proximity_to_minerals,
            accessibility_score=accessibility_score,
            safety_score=safety_score,
            resource_potential=resource_potential,
            environmental_stability=environmental_stability
        )

    def _calculate_solar_exposure(self, slope: float, aspect: float) -> float:
        """Calculate solar exposure based on terrain orientation"""
        # Simplified solar exposure calculation
        # North-facing slopes (aspect ~0°) get less sun in northern hemisphere
        # South-facing slopes (aspect ~180°) get more sun

        aspect_factor = np.abs(np.cos(np.radians(aspect - 180)))  # South-facing optimal
        slope_factor = np.cos(np.radians(slope))  # Less steep = more exposure

        return min(1.0, aspect_factor * slope_factor)

    def _calculate_resource_proximity(self, coordinates: tuple[float, float],
                                    resource_type: str) -> float:
        """Calculate proximity to known Mars resources (simplified model)"""
        lat, lon = coordinates

        # Known resource locations (simplified)
        resource_locations = {
            'water_ice': [
                (85.0, 0.0),    # North polar region
                (-85.0, 0.0),   # South polar region
                (55.0, 150.0),  # Arcadia Planitia
                (32.0, 91.0),   # Protonilus Mensae
            ],
            'minerals': [
                (-15.0, 175.0), # Syrtis Major
                (-18.0, 77.0),  # Arsia Mons
                (0.0, 110.0),   # Arabia Terra
                (-8.0, 282.0),  # Valles Marineris
            ]
        }

        locations = resource_locations.get(resource_type, [])
        if not locations:
            return 0.0

        # Calculate minimum distance to any resource location
        min_distance = float('inf')
        for res_lat, res_lon in locations:
            distance = self._haversine_distance(lat, lon, res_lat, res_lon)
            min_distance = min(min_distance, distance)

        # Convert distance to proximity score (closer = higher score)
        max_distance = 2000  # km - maximum useful distance
        proximity = max(0.0, 1.0 - (min_distance / max_distance))

        return proximity

    def _haversine_distance(self, lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate great-circle distance between two points on Mars"""
        # Mars radius ~3389.5 km
        R = 3389.5

        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))

        return R * c

    def _calculate_accessibility_score(self, terrain_data: dict) -> float:
        """Calculate terrain accessibility score"""
        slope = terrain_data.get('slope_mean', 0.0)
        roughness = terrain_data.get('roughness_mean', 0.0)
        tri = terrain_data.get('tri_mean', 0.0)

        # Lower values are better for accessibility
        slope_score = max(0.0, 1.0 - (slope / 30.0))  # 30° is maximum acceptable slope
        roughness_score = max(0.0, 1.0 - (roughness / 2.0))  # 2m is rough terrain
        tri_score = max(0.0, 1.0 - (tri / 100.0))  # 100 is very rugged

        return (slope_score + roughness_score + tri_score) / 3.0

    def _calculate_safety_score(self, terrain_data: dict) -> float:
        """Calculate terrain safety score"""
        slope = terrain_data.get('slope_mean', 0.0)
        roughness = terrain_data.get('roughness_mean', 0.0)
        elevation = terrain_data.get('elevation', 0.0)

        # Safety factors
        slope_safety = max(0.0, 1.0 - (slope / 25.0))  # 25° max for safety
        roughness_safety = max(0.0, 1.0 - (roughness / 1.5))  # 1.5m max roughness
        elevation_safety = min(1.0, max(0.0, (elevation + 8000) / 10000))  # Prefer higher elevations

        return (slope_safety + roughness_safety + elevation_safety) / 3.0

    def _calculate_resource_potential(self, water_proximity: float, mineral_proximity: float,
                                    terrain_data: dict) -> float:
        """Calculate overall resource potential score"""
        # Weight water ice higher than minerals for survival
        water_weight = 0.7
        mineral_weight = 0.3

        return water_weight * water_proximity + mineral_weight * mineral_proximity

    def _calculate_environmental_stability(self, terrain_data: dict) -> float:
        """Calculate environmental stability score"""
        # Simplified stability based on terrain variation
        slope_std = terrain_data.get('slope_std', 0.0)
        roughness_std = terrain_data.get('roughness_std', 0.0)
        elevation_std = terrain_data.get('elevation_std', 0.0)

        # Lower variation = higher stability
        slope_stability = max(0.0, 1.0 - (slope_std / 10.0))
        roughness_stability = max(0.0, 1.0 - (roughness_std / 0.5))
        elevation_stability = max(0.0, 1.0 - (elevation_std / 100.0))

        return (slope_stability + roughness_stability + elevation_stability) / 3.0

    def prepare_training_data(self, historical_sites: list[dict]) -> pd.DataFrame:
        """
        Prepare training data from historical Mars mission sites

        Args:
            historical_sites: List of dictionaries containing site data and outcomes

        Returns:
            DataFrame with prepared training data
        """
        features_list = []
        scores_list = []

        for site_data in historical_sites:
            # Extract features
            features = self.extract_features_from_terrain_data(site_data['terrain'])

            # Get mission outcome score (0-1 scale)
            outcome_score = site_data.get('mission_success_score', 0.5)

            # Convert features to vector
            feature_vector = [
                features.elevation, features.slope, features.aspect,
                features.roughness, features.tri, features.solar_exposure,
                features.proximity_to_water_ice, features.proximity_to_minerals,
                features.accessibility_score, features.safety_score,
                features.resource_potential, features.environmental_stability
            ]

            features_list.append(feature_vector)
            scores_list.append(outcome_score)

        # Create DataFrame
        feature_names = [
            'elevation', 'slope', 'aspect', 'roughness', 'tri',
            'solar_exposure', 'water_proximity', 'mineral_proximity',
            'accessibility', 'safety', 'resource_potential', 'stability'
        ]

        df = pd.DataFrame(features_list, columns=feature_names)
        df['target_score'] = scores_list

        self.training_data = df
        return df

    def train_models(self, training_data: Optional[pd.DataFrame] = None) -> dict[str, Any]:
        """
        Train ML models using historical site data

        Args:
            training_data: Optional DataFrame with training data

        Returns:
            Dictionary containing training metrics and model performance
        """
        if training_data is None:
            if self.training_data is None:
                raise ValueError("No training data available")
            training_data = self.training_data

        # Prepare features and target
        X = training_data.drop('target_score', axis=1)
        y = training_data['target_score']

        # Scale features
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)

        # Train ensemble model (Random Forest + Gradient Boosting)
        rf_model = RandomForestRegressor(
            n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
        )
        gb_model = GradientBoostingRegressor(
            n_estimators=150, max_depth=10, random_state=42
        )

        # Fit models
        rf_model.fit(X_scaled, y)
        gb_model.fit(X_scaled, y)

        # Create ensemble prediction (average)
        class EnsembleModel:
            def __init__(self, rf_model, gb_model):
                self.rf_model = rf_model
                self.gb_model = gb_model

            def predict(self, X):
                rf_pred = self.rf_model.predict(X)
                gb_pred = self.gb_model.predict(X)
                return (rf_pred + gb_pred) / 2.0

            def predict_with_uncertainty(self, X):
                rf_pred = self.rf_model.predict(X)
                gb_pred = self.gb_model.predict(X)
                ensemble_pred = (rf_pred + gb_pred) / 2.0

                # Calculate uncertainty as standard deviation
                uncertainty = np.std([rf_pred, gb_pred], axis=0)

                return ensemble_pred, uncertainty

        self.ensemble_model = EnsembleModel(rf_model, gb_model)

        # Train clustering model for site categorization
        optimal_clusters = self._find_optimal_clusters(X_scaled)
        self.clustering_model = KMeans(n_clusters=optimal_clusters, random_state=42)
        cluster_labels = self.clustering_model.fit_predict(X_scaled)

        # Calculate clustering metrics
        silhouette_avg = silhouette_score(X_scaled, cluster_labels)
        calinski_harabasz = calinski_harabasz_score(X_scaled, cluster_labels)

        # Feature importance analysis
        feature_importance = rf_model.feature_importances_
        feature_names = X.columns.tolist()
        importance_dict = dict(zip(feature_names, feature_importance, strict=False))

        # Model performance metrics
        from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

        y_pred, uncertainty = self.ensemble_model.predict_with_uncertainty(X_scaled)

        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)

        self.model_metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2_score': r2,
            'silhouette_score': silhouette_avg,
            'calinski_harabasz_score': calinski_harabasz,
            'feature_importance': importance_dict,
            'n_clusters': optimal_clusters,
            'training_samples': len(training_data)
        }

        self.is_trained = True

        # Save models
        self._save_models()

        return self.model_metrics

    def _find_optimal_clusters(self, X: np.ndarray, max_clusters: int = 10) -> int:
        """Find optimal number of clusters using silhouette analysis"""
        silhouette_scores = []

        for n_clusters in range(2, min(max_clusters + 1, len(X))):
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(X)
            silhouette_avg = silhouette_score(X, cluster_labels)
            silhouette_scores.append(silhouette_avg)

        # Return number of clusters with highest silhouette score
        optimal_clusters = silhouette_scores.index(max(silhouette_scores)) + 2
        return optimal_clusters

    def _save_models(self):
        """Save trained models to disk"""
        if not self.is_trained:
            return

        # Save ensemble model components
        joblib.dump(self.ensemble_model.rf_model, self.model_dir / 'rf_model.pkl')
        joblib.dump(self.ensemble_model.gb_model, self.model_dir / 'gb_model.pkl')
        joblib.dump(self.clustering_model, self.model_dir / 'clustering_model.pkl')
        joblib.dump(self.scaler, self.model_dir / 'scaler.pkl')

        # Save metadata
        metadata = {
            'training_date': datetime.now().isoformat(),
            'model_metrics': self.model_metrics,
            'feature_names': [
                'elevation', 'slope', 'aspect', 'roughness', 'tri',
                'solar_exposure', 'water_proximity', 'mineral_proximity',
                'accessibility', 'safety', 'resource_potential', 'stability'
            ]
        }

        with open(self.model_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)

    def load_models(self):
        """Load pre-trained models from disk"""
        try:
            rf_model = joblib.load(self.model_dir / 'rf_model.pkl')
            gb_model = joblib.load(self.model_dir / 'gb_model.pkl')
            clustering_model = joblib.load(self.model_dir / 'clustering_model.pkl')
            scaler = joblib.load(self.model_dir / 'scaler.pkl')

            class EnsembleModel:
                def __init__(self, rf_model, gb_model):
                    self.rf_model = rf_model
                    self.gb_model = gb_model

                def predict(self, X):
                    rf_pred = self.rf_model.predict(X)
                    gb_pred = self.gb_model.predict(X)
                    return (rf_pred + gb_pred) / 2.0

                def predict_with_uncertainty(self, X):
                    rf_pred = self.rf_model.predict(X)
                    gb_pred = self.gb_model.predict(X)
                    ensemble_pred = (rf_pred + gb_pred) / 2.0

                    uncertainty = np.std([rf_pred, gb_pred], axis=0)
                    return ensemble_pred, uncertainty

            self.ensemble_model = EnsembleModel(rf_model, gb_model)
            self.clustering_model = clustering_model
            self.scaler = scaler
            self.is_trained = True

            # Load metadata
            with open(self.model_dir / 'metadata.json') as f:
                metadata = json.load(f)
                self.model_metrics = metadata.get('model_metrics', {})

            return True

        except FileNotFoundError:
            return False

    def recommend_sites(self, candidate_sites: list[dict],
                       mission_type: str = 'research_base',
                       top_n: int = 5) -> list[SiteRecommendation]:
        """
        Generate ML-powered site recommendations

        Args:
            candidate_sites: List of candidate site dictionaries
            mission_type: Type of mission ('research_base', 'mining_operation', etc.)
            top_n: Number of top recommendations to return

        Returns:
            List of SiteRecommendation objects ranked by suitability
        """
        if not self.is_trained:
            raise ValueError("Models must be trained before making recommendations")

        if mission_type not in self.mission_weights:
            mission_type = 'research_base'  # Default mission type

        recommendations = []

        for i, site_data in enumerate(candidate_sites):
            # Extract features
            features = self.extract_features_from_terrain_data(site_data)

            # Convert to feature vector
            feature_vector = [
                features.elevation, features.slope, features.aspect,
                features.roughness, features.tri, features.solar_exposure,
                features.proximity_to_water_ice, features.proximity_to_minerals,
                features.accessibility_score, features.safety_score,
                features.resource_potential, features.environmental_stability
            ]

            # Scale features
            X_scaled = self.scaler.transform([feature_vector])

            # Predict site score with uncertainty
            predicted_score, uncertainty = self.ensemble_model.predict_with_uncertainty(X_scaled)
            predicted_score = predicted_score[0]
            uncertainty = uncertainty[0]

            # Calculate confidence (inverse of uncertainty)
            confidence = max(0.0, 1.0 - uncertainty)

            # Determine cluster assignment
            cluster_assignment = self.clustering_model.predict(X_scaled)[0]

            # Calculate weighted score based on mission type
            weights = self.mission_weights[mission_type]
            weighted_score = self._calculate_weighted_score(features, weights)

            # Combine ML prediction with weighted score
            final_score = 0.7 * predicted_score + 0.3 * weighted_score

            # Generate recommendation reasons and risk factors
            reasons, risks = self._generate_recommendation_reasons(features, weights, final_score)

            recommendation = SiteRecommendation(
                site_id=f"site_{i+1}",
                coordinates=site_data.get('coordinates', (0.0, 0.0)),
                overall_score=final_score,
                feature_scores={
                    'safety': features.safety_score,
                    'accessibility': features.accessibility_score,
                    'resources': features.resource_potential,
                    'stability': features.environmental_stability,
                    'solar_exposure': features.solar_exposure,
                    'terrain': (features.slope + features.roughness + features.tri) / 3.0
                },
                confidence=confidence,
                recommendation_reasons=reasons,
                risk_factors=risks,
                suitability_rank=0,  # Will be set after sorting
                cluster_assignment=cluster_assignment
            )

            recommendations.append(recommendation)

        # Sort by overall score (descending)
        recommendations.sort(key=lambda x: x.overall_score, reverse=True)

        # Update ranks and return top N
        for i, rec in enumerate(recommendations):
            rec.suitability_rank = i + 1

        return recommendations[:top_n]

    def _calculate_weighted_score(self, features: SiteFeatures, weights: dict[str, float]) -> float:
        """Calculate weighted score based on mission-specific criteria"""
        feature_scores = {
            'safety': features.safety_score,
            'accessibility': features.accessibility_score,
            'resources': features.resource_potential,
            'environmental_stability': features.environmental_stability,
            'solar_exposure': features.solar_exposure,
            'terrain': (features.slope + features.roughness + features.tri) / 3.0
        }

        weighted_sum = sum(weights[criterion] * feature_scores[criterion]
                           for criterion in weights)

        return weighted_sum

    def _generate_recommendation_reasons(self, features: SiteFeatures, weights: dict[str, float],
                                       overall_score: float) -> tuple[list[str], list[str]]:
        """Generate human-readable recommendation reasons and risk factors"""
        reasons = []
        risks = []

        feature_scores = {
            'safety': features.safety_score,
            'accessibility': features.accessibility_score,
            'resources': features.resource_potential,
            'stability': features.environmental_stability,
            'solar_exposure': features.solar_exposure,
            'terrain': (features.slope + features.roughness + features.tri) / 3.0
        }

        # Identify top strengths
        sorted_features = sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)

        for feature, score in sorted_features[:3]:  # Top 3 strengths
            if score > 0.7:
                if feature == 'safety':
                    reasons.append("Excellent safety profile with stable terrain conditions")
                elif feature == 'accessibility':
                    reasons.append("High accessibility with gentle slopes and smooth terrain")
                elif feature == 'resources':
                    reasons.append("Strong resource potential with proximity to water ice and minerals")
                elif feature == 'stability':
                    reasons.append("Environmentally stable location with consistent conditions")
                elif feature == 'solar_exposure':
                    reasons.append("Optimal solar exposure for power generation")
                elif feature == 'terrain':
                    reasons.append("Favorable terrain characteristics for construction")

        # Identify risk factors (low scores for high-weight criteria)
        for criterion, weight in weights.items():
            if weight > 0.2 and feature_scores[criterion] < 0.4:
                if criterion == 'safety':
                    risks.append("Low safety score - consider additional safety measures")
                elif criterion == 'accessibility':
                    risks.append("Limited accessibility may complicate logistics")
                elif criterion == 'resources':
                    risks.append("Limited resource availability may require external supply")
                elif criterion == 'environmental_stability':
                    risks.append("Environmental instability could affect long-term operations")

        # Add general risk factors
        if overall_score < 0.5:
            risks.append("Low overall suitability score - consider alternative sites")

        if features.slope > 20:
            risks.append("Steep slopes may pose construction and safety challenges")

        if features.roughness > 1.0:
            risks.append("Rough terrain may require extensive site preparation")

        return reasons, risks

    def get_model_insights(self) -> dict[str, Any]:
        """Get insights about the trained ML models"""
        if not self.is_trained:
            return {"error": "Models not trained yet"}

        return {
            'performance_metrics': self.model_metrics,
            'feature_importance': self.model_metrics.get('feature_importance', {}),
            'clustering_info': {
                'n_clusters': self.model_metrics.get('n_clusters', 0),
                'silhouette_score': self.model_metrics.get('silhouette_score', 0)
            },
            'training_info': {
                'samples': self.model_metrics.get('training_samples', 0),
                'r2_score': self.model_metrics.get('r2_score', 0)
            }
        }

    def update_models_with_feedback(self, site_feedback: list[dict]):
        """
        Update models with user feedback for continuous improvement

        Args:
            site_feedback: List of feedback dictionaries with site_id and rating
        """
        # This would implement online learning to update models with new feedback
        # For now, this is a placeholder for future implementation
        pass
