from utils.grant_finder import Grant, Answers, score_grant, rank_grants


def test_score_prioritizes_org_and_funding():
    g_match = Grant(
        id="1", title="Match", source="X", link="", amount_min=50000, amount_max=100000,
        deadline=None, geographies=["US-National"], org_types=["Nonprofit"], focus_areas=["AI"]
    )
    g_partial = Grant(
        id="2", title="Partial", source="X", link="", amount_min=5000, amount_max=20000,
        deadline=None, geographies=["US-National"], org_types=["Academic"], focus_areas=["AI"]
    )
    ans = Answers(org_type="Nonprofit", funding_min=60000, funding_max=90000,
                  focus_areas=["AI", "Education"], geographies=["US-National"], deadline_within_days=90)

    s1, _ = score_grant(g_match, ans)
    s2, _ = score_grant(g_partial, ans)
    assert s1 > s2, (s1, s2)


def test_rank_orders_by_score():
    grants = [
        Grant(id="1", title="A", source="", link="", amount_min=10000, amount_max=15000, deadline=None,
              geographies=["Global"], org_types=["For-profit"], focus_areas=["Tech"]),
        Grant(id="2", title="B", source="", link="", amount_min=50000, amount_max=80000, deadline=None,
              geographies=["US-National"], org_types=["Nonprofit"], focus_areas=["AI"]),
    ]
    ans = Answers(org_type="Nonprofit", funding_min=60000, funding_max=90000,
                  focus_areas=["AI"], geographies=["US-National"], deadline_within_days=90)

    ranked = rank_grants(grants, ans, top_n=2)
    assert ranked[0]["id"] == "2"
