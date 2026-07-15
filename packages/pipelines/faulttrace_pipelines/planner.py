import math
from typing import List
from faulttrace_core.models import MapPlan, ExtractionUnit
from faulttrace_pipelines.scope_service import ScopeResult

class MapPlanner:
    """
    WP2: Map Planner and Batch Execution.
    Turns the in-scope set into deterministic extraction units (batches).
    """

    @classmethod
    def create_plan(
        cls, 
        run_id: str, 
        world_id: str, 
        scope_result: ScopeResult, 
        batch_size: int = 10
    ) -> MapPlan:
        record_ids = scope_result.eligible_record_ids
        total_eligible = len(record_ids)
        
        units: List[ExtractionUnit] = []
        
        if total_eligible > 0:
            num_batches = math.ceil(total_eligible / batch_size)
            for i in range(num_batches):
                start_idx = i * batch_size
                end_idx = min((i + 1) * batch_size, total_eligible)
                batch_ids = record_ids[start_idx:end_idx]
                
                # We can add token estimation here, but for now we'll rely on the provider
                # or a simple heuristic. 
                token_estimate = len(batch_ids) * 150 # rough estimate
                
                unit = ExtractionUnit(
                    run_id=run_id,
                    record_ids=batch_ids,
                    batch_index=i,
                    token_estimate=token_estimate
                )
                units.append(unit)
                
        plan = MapPlan(
            run_id=run_id,
            world_id=world_id,
            total_eligible_records=total_eligible,
            batch_size=batch_size,
            units=units
        )
        return plan
