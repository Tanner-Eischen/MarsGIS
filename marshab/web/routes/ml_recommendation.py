"""
Machine Learning Site Recommendation API Routes

FastAPI endpoints for ML-powered Mars habitat site recommendation engine.
"""

from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import numpy as np

from marshab.core.ml_site_recommendation import (
    MLSiteRecommendationEngine, SiteRecommendation, SiteFeatures
)
from marshab.core.analysis_pipeline import AnalysisPipeline
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter(prefix="/ml-recommendation", tags=["Machine Learning"])

# Global ML recommendation engine instance
ml_engine = MLSiteRecommendationEngine()

class TrainingDataRequest(BaseModel):
    """Request model for ML model training"""
    historical_sites: List[Dict[str, Any]] = Field(..., description="Historical Mars mission sites with outcomes")
    mission_type: str = Field(default="research_base", description="Mission type for training")
    
class TrainingDataResponse(BaseModel):
    """Response model for ML model training"""
    success: bool
    message: str
    training_metrics: Dict[str, Any]
    model_performance: Dict[str, Any]
    
class SiteRecommendationRequest(BaseModel):
    """Request model for site recommendations"""
    candidate_sites: List[Dict[str, Any]] = Field(..., description="Candidate sites for recommendation")
    mission_type: str = Field(default="research_base", description="Type of mission")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")
    use_predefined_weights: bool = Field(default=True, description="Use predefined mission weights")
    
class SiteRecommendationResponse(BaseModel):
    """Response model for site recommendations"""
    success: bool
    recommendations: List[Dict[str, Any]]
    mission_type: str
    total_candidates: int
    recommendation_count: int
    processing_time: float
    model_confidence: float
    
class ModelInsightsResponse(BaseModel):
    """Response model for model insights"""
    success: bool
    model_status: str
    performance_metrics: Dict[str, Any]
    feature_importance: Dict[str, str]
    clustering_info: Dict[str, Any]
    training_info: Dict[str, Any]
    
class BatchRecommendationRequest(BaseModel):
    """Request model for batch site recommendations"""
    regions: List[Dict[str, Any]] = Field(..., description="Regions to analyze")
    mission_types: List[str] = Field(default=["research_base"], description="Mission types to consider")
    analysis_depth: str = Field(default="standard", description="Analysis depth: quick, standard, comprehensive")
    
class BatchRecommendationResponse(BaseModel):
    """Response model for batch recommendations"""
    success: bool
    regional_recommendations: Dict[str, List[Dict[str, Any]]]
    analysis_summary: Dict[str, Any]
    processing_time: float
    
# Predefined training data for demonstration
SAMPLE_TRAINING_DATA = [
    {
        "terrain": {
            "elevation": -2000,
            "slope_mean": 5.2,
            "aspect_mean": 135.0,
            "roughness_mean": 0.3,
            "tri_mean": 15.0,
            "coordinates": (18.0, 77.0),
            "slope_std": 2.1,
            "roughness_std": 0.1,
            "elevation_std": 50.0
        },
        "mission_success_score": 0.85,
        "mission_type": "research_base",
        "site_name": "Olympus Mons Base Alpha"
    },
    {
        "terrain": {
            "elevation": -4000,
            "slope_mean": 12.5,
            "aspect_mean": 45.0,
            "roughness_mean": 0.8,
            "tri_mean": 35.0,
            "coordinates": (-15.0, 175.0),
            "slope_std": 5.2,
            "roughness_std": 0.3,
            "elevation_std": 120.0
        },
        "mission_success_score": 0.72,
        "mission_type": "mining_operation",
        "site_name": "Syrtis Major Mining Outpost"
    },
    {
        "terrain": {
            "elevation": 1000,
            "slope_mean": 2.1,
            "aspect_mean": 180.0,
            "roughness_mean": 0.1,
            "tri_mean": 8.0,
            "coordinates": (32.0, 91.0),
            "slope_std": 0.8,
            "roughness_std": 0.05,
            "elevation_std": 25.0
        },
        "mission_success_score": 0.92,
        "mission_type": "emergency_shelter",
        "site_name": "Protonilus Emergency Station"
    },
    {
        "terrain": {
            "elevation": -1500,
            "slope_mean": 8.7,
            "aspect_mean": 90.0,
            "roughness_mean": 0.5,
            "tri_mean": 22.0,
            "coordinates": (55.0, 150.0),
            "slope_std": 3.1,
            "roughness_std": 0.2,
            "elevation_std": 80.0
        },
        "mission_success_score": 0.78,
        "mission_type": "permanent_settlement",
        "site_name": "Arcadia Planitia Colony"
    },
    {
        "terrain": {
            "elevation": -3500,
            "slope_mean": 15.2,
            "aspect_mean": 270.0,
            "roughness_mean": 1.2,
            "tri_mean": 45.0,
            "coordinates": (-8.0, 282.0),
            "slope_std": 7.8,
            "roughness_std": 0.6,
            "elevation_std": 200.0
        },
        "mission_success_score": 0.45,
        "mission_type": "research_base",
        "site_name": "Valles Marineris Research Site"
    },
    {
        "terrain": {
            "elevation": 3000,
            "slope_mean": 1.8,
            "aspect_mean": 200.0,
            "roughness_mean": 0.2,
            "tri_mean": 12.0,
            "coordinates": (0.0, 110.0),
            "slope_std": 1.2,
            "roughness_std": 0.08,
            "elevation_std": 40.0
        },
        "mission_success_score": 0.88,
        "mission_type": "mining_operation",
        "site_name": "Arabia Terra Mining Complex"
    }
]

@router.post("/train", response_model=TrainingDataResponse)
async def train_ml_models(request: TrainingDataRequest):
    """
    Train ML models using historical Mars mission data
    
    This endpoint trains the machine learning models using historical
    Mars mission site data and their success outcomes.
    """
    try:
        start_time = datetime.now()
        
        # Use provided training data or sample data
        training_sites = request.historical_sites if request.historical_sites else SAMPLE_TRAINING_DATA
        
        logger.info(f"Training ML models with {len(training_sites)} historical sites")
        
        # Prepare training data
        ml_engine.prepare_training_data(training_sites)
        
        # Train models
        training_metrics = ml_engine.train_models()
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return TrainingDataResponse(
            success=True,
            message=f"ML models trained successfully with {len(training_sites)} samples",
            training_metrics=training_metrics,
            model_performance={
                "r2_score": training_metrics.get("r2_score", 0),
                "rmse": training_metrics.get("rmse", 0),
                "silhouette_score": training_metrics.get("silhouette_score", 0),
                "training_samples": training_metrics.get("training_samples", 0)
            }
        )
        
    except Exception as e:
        logger.error(f"ML training failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"ML training failed: {str(e)}")

@router.post("/recommend", response_model=SiteRecommendationResponse)
async def get_site_recommendations(request: SiteRecommendationRequest):
    """
    Get ML-powered site recommendations for Mars habitat placement
    
    This endpoint uses trained machine learning models to recommend
    the most suitable sites for Mars habitat placement based on
    terrain characteristics and mission requirements.
    """
    try:
        start_time = datetime.now()
        
        # Check if models are trained
        if not ml_engine.is_trained:
            logger.info("Models not trained, using sample training data")
            # Auto-train with sample data
            ml_engine.prepare_training_data(SAMPLE_TRAINING_DATA)
            ml_engine.train_models()
        
        logger.info(f"Generating recommendations for {len(request.candidate_sites)} candidate sites")
        logger.info(f"Mission type: {request.mission_type}, Top N: {request.top_n}")
        
        # Generate recommendations
        recommendations = ml_engine.recommend_sites(
            candidate_sites=request.candidate_sites,
            mission_type=request.mission_type,
            top_n=request.top_n
        )
        
        # Convert recommendations to serializable format
        serializable_recommendations = []
        total_confidence = 0.0
        
        for rec in recommendations:
            rec_dict = {
                "site_id": rec.site_id,
                "coordinates": rec.coordinates,
                "overall_score": round(rec.overall_score, 3),
                "feature_scores": {k: round(v, 3) for k, v in rec.feature_scores.items()},
                "confidence": round(rec.confidence, 3),
                "recommendation_reasons": rec.recommendation_reasons,
                "risk_factors": rec.risk_factors,
                "suitability_rank": rec.suitability_rank,
                "cluster_assignment": rec.cluster_assignment,
                "suitability_category": _get_suitability_category(rec.overall_score)
            }
            serializable_recommendations.append(rec_dict)
            total_confidence += rec.confidence
        
        avg_confidence = total_confidence / len(recommendations) if recommendations else 0.0
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return SiteRecommendationResponse(
            success=True,
            recommendations=serializable_recommendations,
            mission_type=request.mission_type,
            total_candidates=len(request.candidate_sites),
            recommendation_count=len(recommendations),
            processing_time=processing_time,
            model_confidence=round(avg_confidence, 3)
        )
        
    except Exception as e:
        logger.error(f"Site recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Site recommendation failed: {str(e)}")

@router.get("/insights", response_model=ModelInsightsResponse)
async def get_model_insights():
    """
    Get insights about the trained ML models
    
    This endpoint provides detailed information about the
    trained machine learning models, including performance
    metrics and feature importance analysis.
    """
    try:
        # Check if models are trained
        if not ml_engine.is_trained:
            return ModelInsightsResponse(
                success=True,
                model_status="not_trained",
                performance_metrics={},
                feature_importance={},
                clustering_info={},
                training_info={}
            )
        
        insights = ml_engine.get_model_insights()
        
        # Format feature importance as percentages
        feature_importance = {}
        importance_data = insights.get('feature_importance', {})
        if importance_data:
            total_importance = sum(importance_data.values())
            for feature, importance in importance_data.items():
                percentage = (importance / total_importance) * 100 if total_importance > 0 else 0
                feature_importance[feature] = f"{percentage:.1f}%"
        
        return ModelInsightsResponse(
            success=True,
            model_status="trained",
            performance_metrics=insights.get('performance_metrics', {}),
            feature_importance=feature_importance,
            clustering_info=insights.get('clustering_info', {}),
            training_info=insights.get('training_info', {})
        )
        
    except Exception as e:
        logger.error(f"Failed to get model insights: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Failed to get model insights: {str(e)}")

@router.post("/batch-analyze", response_model=BatchRecommendationResponse)
async def batch_site_analysis(request: BatchRecommendationRequest, background_tasks: BackgroundTasks):
    """
    Perform batch analysis across multiple regions
    
    This endpoint performs comprehensive ML analysis across multiple
    regions to identify optimal habitat placement opportunities.
    """
    try:
        start_time = datetime.now()
        
        logger.info(f"Starting batch analysis for {len(request.regions)} regions")
        logger.info(f"Mission types: {request.mission_types}")
        logger.info(f"Analysis depth: {request.analysis_depth}")
        
        # Ensure models are trained
        if not ml_engine.is_trained:
            ml_engine.prepare_training_data(SAMPLE_TRAINING_DATA)
            ml_engine.train_models()
        
        regional_recommendations = {}
        analysis_summary = {
            "total_regions_analyzed": len(request.regions),
            "mission_types_considered": request.mission_types,
            "analysis_depth": request.analysis_depth,
            "optimal_sites_found": 0,
            "high_confidence_sites": 0
        }
        
        for region in request.regions:
            region_name = region.get("name", "Unknown Region")
            
            # Generate candidate sites for this region (simplified)
            candidate_sites = _generate_region_candidates(region)
            
            region_recs = []
            for mission_type in request.mission_types:
                recommendations = ml_engine.recommend_sites(
                    candidate_sites=candidate_sites,
                    mission_type=mission_type,
                    top_n=3  # Top 3 per mission type
                )
                
                for rec in recommendations:
                    rec_data = {
                        "mission_type": mission_type,
                        "site_id": rec.site_id,
                        "coordinates": rec.coordinates,
                        "overall_score": round(rec.overall_score, 3),
                        "confidence": round(rec.confidence, 3),
                        "suitability_rank": rec.suitability_rank,
                        "suitability_category": _get_suitability_category(rec.overall_score)
                    }
                    region_recs.append(rec_data)
            
            # Sort by score and take top recommendations
            region_recs.sort(key=lambda x: x["overall_score"], reverse=True)
            regional_recommendations[region_name] = region_recs[:5]
            
            # Update summary statistics
            analysis_summary["optimal_sites_found"] += len(region_recs)
            analysis_summary["high_confidence_sites"] += sum(1 for rec in region_recs if rec["confidence"] > 0.8)
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        return BatchRecommendationResponse(
            success=True,
            regional_recommendations=regional_recommendations,
            analysis_summary=analysis_summary,
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Batch analysis failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Batch analysis failed: {str(e)}")

def _get_suitability_category(score: float) -> str:
    """Convert numerical score to suitability category"""
    if score >= 0.8:
        return "Excellent"
    elif score >= 0.6:
        return "Good"
    elif score >= 0.4:
        return "Fair"
    else:
        return "Poor"

def _generate_region_candidates(region: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Generate candidate sites for a region (simplified)"""
    # This would typically use actual terrain data from the region
    # For now, generate synthetic candidates based on region bounds
    
    bounds = region.get("bounds", {})
    lat_min = bounds.get("lat_min", -10)
    lat_max = bounds.get("lat_max", 10)
    lon_min = bounds.get("lon_min", 0)
    lon_max = bounds.get("lon_max", 20)
    
    candidates = []
    for i in range(9):  # 3x3 grid
        lat = lat_min + (lat_max - lat_min) * (i // 3) / 2
        lon = lon_min + (lon_max - lon_min) * (i % 3) / 2
        
        # Generate varied terrain characteristics
        base_elevation = np.random.uniform(-4000, 2000)
        base_slope = np.random.uniform(1, 15)
        base_roughness = np.random.uniform(0.1, 1.0)
        
        candidate = {
            "coordinates": (lat, lon),
            "elevation": base_elevation,
            "slope_mean": base_slope,
            "aspect_mean": np.random.uniform(0, 360),
            "roughness_mean": base_roughness,
            "tri_mean": np.random.uniform(5, 40),
            "slope_std": base_slope * 0.3,
            "roughness_std": base_roughness * 0.2,
            "elevation_std": np.random.uniform(20, 150)
        }
        candidates.append(candidate)
    
    return candidates

@router.get("/health")
async def health_check():
    """Health check endpoint for ML recommendation service"""
    return {
        "status": "healthy",
        "service": "ml_site_recommendation",
        "models_trained": ml_engine.is_trained,
        "timestamp": datetime.now().isoformat()
    }
class DestinationAwareRecommendationRequest(BaseModel):
    destination_coordinates: Tuple[float, float] = Field(..., description="Destination site coordinates [lat, lon]")
    candidate_sites: List[Dict[str, Any]] = Field(..., description="Candidate landing sites")
    mission_type: str = Field(default="research_base", description="Type of mission")
    top_n: int = Field(default=5, ge=1, le=20, description="Number of recommendations to return")

@router.post("/recommend-for-destination", response_model=SiteRecommendationResponse)
async def get_destination_aware_recommendations(request: DestinationAwareRecommendationRequest):
    try:
        start_time = datetime.now()
        if not ml_engine.is_trained:
            ml_engine.prepare_training_data(SAMPLE_TRAINING_DATA)
            ml_engine.train_models()

        base_recs = ml_engine.recommend_sites(
            candidate_sites=request.candidate_sites,
            mission_type=request.mission_type,
            top_n=max(request.top_n, len(request.candidate_sites))
        )

        dlat, dlon = request.destination_coordinates

        def haversine_km(lat1, lon1, lat2, lon2):
            R = 3389.5
            phi1 = np.deg2rad(lat1)
            phi2 = np.deg2rad(lat2)
            dphi = np.deg2rad(lat2 - lat1)
            dlambda = np.deg2rad(lon2 - lon1)
            a = np.sin(dphi/2)**2 + np.cos(phi1)*np.cos(phi2)*np.sin(dlambda/2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            return R * c

        distances = []
        for c in request.candidate_sites:
            clat, clon = c.get("coordinates", (0.0, 0.0))
            distances.append(haversine_km(clat, clon, dlat, dlon))
        max_dist = max(distances) if distances else 1.0

        if request.mission_type == "emergency_shelter":
            w_d, w_r = 0.6, 0.5
        elif request.mission_type == "permanent_settlement":
            w_d, w_r = 0.3, 0.4
        elif request.mission_type == "mining_operation":
            w_d, w_r = 0.2, 0.3
        else:
            w_d, w_r = 0.15, 0.25

        indexed_candidates = {}
        for c in request.candidate_sites:
            key = tuple(c.get("coordinates", (0.0, 0.0)))
            indexed_candidates[key] = c

        enriched = []
        for rec in base_recs:
            lat, lon = rec.coordinates
            cand = indexed_candidates.get((lat, lon), {})
            dist_km = haversine_km(lat, lon, dlat, dlon)
            dist_norm = dist_km / max_dist if max_dist > 0 else 0.0
            slope = float(cand.get("slope_mean", 0.0))
            rough = float(cand.get("roughness_mean", 0.0))
            risk_norm = ((slope / 30.0) + (rough / 2.0)) / 2.0
            adj = float(rec.overall_score) - (w_d * dist_norm + w_r * risk_norm)
            adj = max(0.0, min(1.0, adj))
            enriched.append((rec, adj, dist_km, risk_norm))

        enriched.sort(key=lambda x: x[1], reverse=True)
        top = enriched[:request.top_n]

        serializable = []
        total_conf = 0.0
        for rank_idx, (rec, adj, dist_km, risk_norm) in enumerate(top, start=1):
            rec_dict = {
                "site_id": rec.site_id,
                "coordinates": rec.coordinates,
                "overall_score": round(float(rec.overall_score), 3),
                "adjusted_score": round(float(adj), 3),
                "feature_scores": {k: round(v, 3) for k, v in rec.feature_scores.items()},
                "confidence": round(float(rec.confidence), 3),
                "recommendation_reasons": rec.recommendation_reasons + ["Proximity to destination"],
                "risk_factors": rec.risk_factors + [f"Estimated relative route cost: {dist_km:.1f} km, risk {risk_norm:.2f}"],
                "suitability_rank": rank_idx,
                "cluster_assignment": rec.cluster_assignment,
                "suitability_category": _get_suitability_category(float(adj))
            }
            serializable.append(rec_dict)
            total_conf += float(rec.confidence)

        avg_conf = total_conf / len(top) if top else 0.0
        processing_time = (datetime.now() - start_time).total_seconds()

        return SiteRecommendationResponse(
            success=True,
            recommendations=serializable,
            mission_type=request.mission_type,
            total_candidates=len(request.candidate_sites),
            recommendation_count=len(serializable),
            processing_time=processing_time,
            model_confidence=round(avg_conf, 3)
        )
    except Exception as e:
        logger.error(f"Destination-aware recommendation failed: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Destination-aware recommendation failed: {str(e)}")
