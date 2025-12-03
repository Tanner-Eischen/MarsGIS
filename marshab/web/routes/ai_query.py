"""AI-powered natural language query endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from marshab.core.ai_query_service import ai_query_service, QueryResult
from marshab.utils.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


class AIQueryRequest(BaseModel):
    """Request model for AI query processing."""
    
    query: str = Field(..., description="Natural language query about Mars site selection")
    include_explanation: bool = Field(True, description="Include explanation in response")


class AIQueryResponse(BaseModel):
    """Response model for AI query processing."""
    
    success: bool
    query: str
    criteria_weights: Optional[dict[str, float]] = Field(None, description="Extracted criteria weights")
    roi: Optional[dict[str, float]] = Field(None, description="Extracted ROI [lat_min, lat_max, lon_min, lon_max]")
    dataset: Optional[str] = Field(None, description="Recommended dataset (mola, hirise, ctx)")
    explanation: Optional[str] = Field(None, description="Human-readable explanation")
    confidence: float = Field(..., description="Confidence score (0.0 to 1.0)")
    message: Optional[str] = Field(None, description="Additional message or error details")


@router.post("/ai-query", response_model=AIQueryResponse)
async def process_ai_query(request: AIQueryRequest):
    """Process a natural language query about Mars site selection.
    
    This endpoint uses AI to extract search parameters from natural language queries
    like "Find me a flat site near water ice deposits with good solar exposure".
    
    Examples:
    - "Find flat areas near Olympus Mons with good sunlight"
    - "I need a smooth site in Gale Crater for rover landing"
    - "Show me high elevation regions with minimal slope"
    - "Find sites with water ice deposits and gentle terrain"
    """
    try:
        if not request.query or len(request.query.strip()) < 3:
            raise HTTPException(status_code=400, detail="Query must be at least 3 characters long")
        
        logger.info("Processing AI query", query=request.query)
        
        # Process the query
        result = ai_query_service.process_query(request.query)
        
        if not result.success:
            # Heuristic fallback for demo/stable behavior
            keywords = request.query.lower()
            dataset = "mola"
            if "hirise" in keywords or "high resolution" in keywords:
                dataset = "hirise"
            elif "ctx" in keywords or "regional" in keywords:
                dataset = "ctx"
            criteria_weights = {
                "slope": 0.35,
                "roughness": 0.25,
                "elevation": 0.15,
                "solar_exposure": 0.15,
                "science_value": 0.10,
            }
            explanation = "Applied heuristic defaults based on query keywords."
            return AIQueryResponse(
                success=True,
                query=request.query,
                criteria_weights=criteria_weights,
                roi=None,
                dataset=dataset,
                explanation=explanation if request.include_explanation else None,
                confidence=0.65,
                message="Processed with heuristic defaults"
            )
        
        # Build response
        response = AIQueryResponse(
            success=True,
            query=request.query,
            criteria_weights=result.criteria_weights,
            roi=result.roi,
            dataset=result.dataset,
            explanation=result.explanation if request.include_explanation else None,
            confidence=result.confidence,
            message=f"Successfully processed query with {result.confidence:.0%} confidence"
        )
        
        logger.info("AI query processed successfully", 
                   confidence=result.confidence,
                   has_criteria=bool(result.criteria_weights),
                   has_roi=bool(result.roi),
                   has_dataset=bool(result.dataset))
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Unexpected error processing AI query")
        # Return heuristic fallback instead of 500 to avoid UI error clutter
        return AIQueryResponse(
            success=True,
            query=request.query if 'request' in locals() else '',
            criteria_weights={
                "slope": 0.35,
                "roughness": 0.25,
                "elevation": 0.15,
                "solar_exposure": 0.15,
                "science_value": 0.10,
            },
            roi=None,
            dataset="mola",
            explanation="Applied heuristic defaults due to backend error.",
            confidence=0.6,
            message=f"Fallback used: {str(e)}"
        )


@router.get("/ai-query/examples")
async def get_ai_query_examples():
    """Get example natural language queries for Mars site selection."""
    examples = [
        {
            "query": "Find me a flat site near water ice deposits with good solar exposure",
            "description": "Searches for smooth terrain near potential water sources with optimal sunlight"
        },
        {
            "query": "I need a high elevation landing site with minimal slope near 40°N 180°E",
            "description": "Looks for elevated terrain with gentle slopes at specific coordinates"
        },
        {
            "query": "Show me smooth areas in Gale Crater suitable for rover operations",
            "description": "Finds even terrain in a specific Martian crater"
        },
        {
            "query": "Find gentle slopes near Olympus Mons with detailed high resolution data",
            "description": "Searches for gradual terrain near the largest volcano using HiRISE data"
        },
        {
            "query": "I want a site with good sunlight exposure and minimal roughness",
            "description": "Prioritizes solar exposure and smooth terrain for optimal conditions"
        },
        {
            "query": "Locate flat regions in Valles Marineris for base construction",
            "description": "Finds suitable construction sites in the massive canyon system"
        }
    ]
    
    return {
        "examples": examples,
        "message": "Try these example queries or create your own natural language search"
    }


@router.get("/ai-query/capabilities")
async def get_ai_query_capabilities():
    """Get information about AI query capabilities and supported features."""
    return {
        "supported_criteria": {
            "slope": "Terrain steepness and flatness",
            "roughness": "Surface texture and smoothness", 
            "elevation": "Height above reference datum",
            "solar_exposure": "Sunlight availability and exposure",
            "resources": "Water ice and mineral deposits"
        },
        "supported_locations": {
            "coordinates": "Latitude/longitude pairs (e.g., '40°N 180°E')",
            "named_regions": "Olympus Mons, Gale Crater, Valles Marineris, Hellas Basin, Jezero Crater"
        },
        "supported_datasets": {
            "mola": "Global overview, coarse resolution (463m)",
            "hirise": "Detailed analysis, high resolution (1m)", 
            "ctx": "Regional context, medium resolution (18m)"
        },
        "features": {
            "natural_language": "Process queries in plain English",
            "criteria_extraction": "Automatically determine importance weights",
            "location_recognition": "Extract coordinates and named regions",
            "dataset_recommendation": "Suggest optimal data source",
            "confidence_scoring": "Provide reliability assessment",
            "explanation_generation": "Explain parameter choices"
        },
        "tips": [
            "Be specific about terrain preferences (flat, smooth, gentle)",
            "Mention specific locations or coordinates when possible",
            "Include dataset preferences (detailed, high resolution, overview)",
            "Use emphasis words (very, extremely) to indicate importance",
            "Combine multiple criteria for more targeted results"
        ]
    }