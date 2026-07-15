"""Demo seeding endpoint."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException
from sqlalchemy.orm import Session

from faulttrace_api.config import get_settings
from faulttrace_api.database import get_db, WorldRow, QueryRow
from faulttrace_api.models import SeedDemoRequest

router = APIRouter()


@router.post("/demo/seed", summary="Seed deterministic demo data")
async def seed_demo(
    request: SeedDemoRequest,
    db: Session = Depends(get_db),
):
    """
    Generate deterministic nested corpus worlds and procedural queries.
    
    Creates worlds at the specified scales from a fixed seed.
    All generated data is deterministic and reproducible.
    """
    settings = get_settings()
    data_root = settings.data_root
    
    # Check if already seeded
    existing = db.query(WorldRow).filter(WorldRow.seed == request.seed).count()
    if existing > 0 and not request.overwrite:
        worlds = db.query(WorldRow).filter(WorldRow.seed == request.seed).all()
        return {
            "status": "already_seeded",
            "seed": request.seed,
            "world_ids": [w.world_id for w in worlds],
            "message": "Pass overwrite=true to re-seed",
        }
    
    from faulttrace_data.generator import TrackMGenerator
    from faulttrace_pipelines.query_factory import QueryFactory
    from faulttrace_gold.validator import GoldValidator
    import pandas as pd
    
    generator = TrackMGenerator(seed=request.seed)
    worlds_dir = data_root / "generated" / "worlds"
    
    try:
        # Generate worlds
        results = generator.generate_nested_worlds(
            scales=sorted(request.scales),
            output_dir=worlds_dir,
        )
        
        world_ids = []
        for world, manifest in results:
            # Upsert world row
            existing_row = db.query(WorldRow).filter(WorldRow.world_id == world.world_id).first()
            if existing_row:
                db.delete(existing_row)
            
            row = WorldRow(
                world_id=world.world_id,
                dataset_id=world.dataset_id,
                seed=world.seed,
                scale_n=world.scale_n,
                parent_world_id=world.parent_world_id,
                creation_policy=world.creation_policy,
                record_ids_hash=world.record_ids_hash,
                manifest_path=world.manifest_path,
                created_at=world.created_at.replace(tzinfo=None),
                schema_version=world.schema_version,
            )
            db.add(row)
            world_ids.append(world.world_id)
        
        db.commit()
        
        # Generate queries for the largest world
        largest_world_id = world_ids[-1]
        parquet_path = worlds_dir / largest_world_id / "records.parquet"
        df = pd.read_parquet(parquet_path)
        
        factory = QueryFactory(data_dir=data_root / "generated")
        validator = GoldValidator()
        queries = factory.generate_for_world(world_id=largest_world_id, target_count=60)
        
        # Validate and store queries
        queries_stored = 0
        queries_dir = settings.artifacts_root / "queries"
        queries_dir.mkdir(parents=True, exist_ok=True)
        
        out_path = queries_dir / f"queries_{largest_world_id}.jsonl"
        with open(out_path, "w") as f:
            for q in queries:
                # Compute gold answer
                gold_result = validator.validate(q, df, parquet_path)
                
                # Store in DB
                existing_q = db.query(QueryRow).filter(QueryRow.query_id == q.query_id).first()
                if existing_q:
                    db.delete(existing_q)
                
                gold_json = None
                if gold_result.gold_answer:
                    gold_json = json.dumps(gold_result.gold_answer.model_dump(mode="json"), default=str)
                
                q_row = QueryRow(
                    query_id=q.query_id,
                    world_id=q.world_id,
                    family=q.family.value,
                    natural_language_question=q.natural_language_question,
                    template_id=q.template_id,
                    version=q.version,
                    spec_json=json.dumps(q.model_dump(mode="json"), default=str),
                    gold_json=gold_json,
                    created_at=q.created_at.replace(tzinfo=None),
                )
                db.add(q_row)
                f.write(json.dumps(q.model_dump(mode="json"), default=str) + "\n")
                queries_stored += 1
        
        db.commit()
        
        # Generate adversarial fixtures
        fixture_dir = data_root / "generated" / "fixtures"
        generator.generate_adversarial_fixtures(fixture_dir)
        
        return {
            "status": "seeded",
            "seed": request.seed,
            "world_ids": world_ids,
            "queries_generated": queries_stored,
            "largest_world_id": largest_world_id,
            "data_root": str(data_root),
        }
    
    except Exception as e:
        db.rollback()
        raise HTTPException(status_code=500, detail=f"Seeding failed: {e}")
