import re
from difflib import SequenceMatcher
from typing import Dict, List, Optional, Tuple


def _normalize(text: str) -> str:
    return (text or "").strip().lower().replace("ё", "е")


def _tokenize(text: str) -> List[str]:
    return re.findall(r"[a-zA-Zа-яА-Я0-9_]+", _normalize(text))


def _phrase_in_text(phrase: str, text: str) -> bool:
    p = _normalize(phrase)
    t = _normalize(text)
    if not p or not t:
        return False
    return p in t


def _text_similarity(a: str, b: str) -> float:
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


def _build_alias_map(items: Dict[str, dict], synonyms: Dict[str, str]) -> Dict[str, List[str]]:
    alias_map: Dict[str, List[str]] = {}
    for name, cfg in items.items():
        aliases = cfg.get("aliases", []) if isinstance(cfg, dict) else []
        collected = {name}
        for alias in aliases:
            if isinstance(alias, str) and alias.strip():
                collected.add(alias.strip())

        for src, target in synonyms.items():
            if _normalize(target) == _normalize(name):
                collected.add(src)

        alias_map[name] = sorted(collected)
    return alias_map


def _resolve_entity(
    query: str,
    items: Dict[str, dict],
    synonyms: Dict[str, str],
    fuzzy_threshold: float = 0.7,
) -> Tuple[Optional[str], List[dict], str]:
    if not items:
        return None, [], "none"

    query_norm = _normalize(query)
    alias_map = _build_alias_map(items, synonyms)
    scored: List[dict] = []

    for name, aliases in alias_map.items():
        best_score = 0.0
        best_match = ""
        match_type = "none"

        for alias in aliases:
            alias_norm = _normalize(alias)
            if not alias_norm:
                continue

            if _phrase_in_text(alias_norm, query_norm):
                local_score = 1.0
                local_type = "exact"
            else:
                token_scores = [_text_similarity(token, alias_norm) for token in _tokenize(query_norm)]
                phrase_score = _text_similarity(query_norm, alias_norm)
                local_score = max(token_scores + [phrase_score]) if token_scores else phrase_score
                local_type = "fuzzy" if local_score >= fuzzy_threshold else "none"

            if local_score > best_score:
                best_score = local_score
                best_match = alias
                match_type = local_type

        if best_score > 0:
            scored.append(
                {
                    "name": name,
                    "score": round(best_score, 3),
                    "match": best_match,
                    "match_type": match_type,
                }
            )

    scored.sort(key=lambda x: x["score"], reverse=True)

    if not scored:
        return None, [], "none"

    top = scored[0]
    resolved = top["name"] if top["match_type"] in {"exact", "fuzzy"} else None
    if top["match_type"] == "fuzzy" and top["score"] < fuzzy_threshold:
        resolved = None

    ambiguities = []
    if resolved:
        for alt in scored[1:4]:
            if alt["score"] >= max(0.65, top["score"] - 0.08):
                ambiguities.append(
                    {
                        "type": "entity",
                        "candidate": alt["name"],
                        "score": alt["score"],
                        "description": f"Возможно, вы имели в виду: {alt['name']}",
                    }
                )

    return resolved, ambiguities, top["match_type"]


def _resolve_time_expression(semantic_layer: dict, user_query: str) -> Optional[str]:
    q = _normalize(user_query)
    for phrase in semantic_layer.get("time_expressions", {}).keys():
        if _phrase_in_text(phrase, q):
            return phrase
    return None


def _resolve_filters(semantic_layer: dict, user_query: str) -> List[str]:
    q = _normalize(user_query)
    filters = semantic_layer.get("filters", {})
    synonyms = semantic_layer.get("synonyms", {})
    found: List[str] = []

    for name, cfg in filters.items():
        aliases = [name]
        if isinstance(cfg, dict):
            aliases.extend(cfg.get("aliases", []))

        for src, target in synonyms.items():
            if _normalize(target) == _normalize(name):
                aliases.append(src)

        if any(_phrase_in_text(alias, q) for alias in aliases if isinstance(alias, str)):
            found.append(name)

    return sorted(set(found))


def resolve_intent_with_confidence(user_query: str, semantic_layer: dict) -> dict:
    synonyms = semantic_layer.get("synonyms", {})

    metric, metric_amb, metric_match_type = _resolve_entity(
        query=user_query,
        items=semantic_layer.get("metrics", {}),
        synonyms=synonyms,
    )
    dimension, dimension_amb, dimension_match_type = _resolve_entity(
        query=user_query,
        items=semantic_layer.get("dimensions", {}),
        synonyms=synonyms,
    )

    time_range = _resolve_time_expression(semantic_layer, user_query)
    filters = _resolve_filters(semantic_layer, user_query)

    confidence = 0.0
    if metric:
        confidence += 0.4 if metric_match_type == "exact" else 0.3
    if dimension:
        confidence += 0.4 if dimension_match_type == "exact" else 0.3
    if time_range:
        confidence += 0.2

    ambiguities = []
    for amb in metric_amb:
        ambiguities.append({
            "kind": "metric",
            "value": amb["candidate"],
            "description": amb["description"],
            "score": amb["score"],
        })
    for amb in dimension_amb:
        ambiguities.append({
            "kind": "dimension",
            "value": amb["candidate"],
            "description": amb["description"],
            "score": amb["score"],
        })

    if ambiguities:
        confidence = max(0.0, confidence - 0.15)

    confidence = max(0.0, min(1.0, confidence))

    return {
        "metric": metric,
        "dimension": dimension,
        "time_range": time_range,
        "filters": filters,
        "confidence": round(confidence, 2),
        "ambiguities": ambiguities,
    }
