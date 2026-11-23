trades_changed = False
for trade in st.session_state.active_trades:
    coin = trade["coin"]
    tf = trade["timeframe"]
    direction = trade["pattern_type"]
    live_price = last_price_by_coin.get(coin)
    if live_price is None:
        live_price = trade.get("last_price")
    trade["last_price"] = live_price

    latest_scan = result_map.get((coin, tf))
    if latest_scan:
        trade["conf_score"] = latest_scan.conf_score
        trade["conf_pct"] = latest_scan.conf_pct
        trade["total_score"] = latest_scan.total_score
        trade["bad_confluences"] = latest_scan.bad_confluences
        trade["beyond_optimal_entry"] = latest_scan.beyond_optimal_entry
        trade["optimal_entry_status"] = latest_scan.optimal_entry_status
        trade["secondary_sl"] = latest_scan.secondary_sl
        trade["bias_change_level"] = latest_scan.bias_change_level
        trade["mtf_notes"] = latest_scan.mtf_notes
        trade["tp_obstacle_note"] = latest_scan.tp_obstacle_note

    if live_price is not None and not trade.get("sl_hit", False):
        if direction == "BULLISH":
            if trade.get("tp1") is not None and live_price >= trade["tp1"] and not trade.get("tp1_hit"):
                trade["tp1_hit"] = True
                trades_changed = True
            if trade.get("tp2") is not None and live_price >= trade["tp2"] and not trade.get("tp2_hit"):
                trade["tp2_hit"] = True
                trades_changed = True
            if trade.get("tp3") is not None and live_price >= trade["tp3"] and not trade.get("tp3_hit"):
                trade["tp3_hit"] = True
                trades_changed = True
            if trade.get("sl") is not None and live_price <= trade["sl"]:
                trade["sl_hit"] = True
                trades_changed = True
        else:
            if trade.get("tp1") is not None and live_price <= trade["tp1"] and not trade.get("tp1_hit"):
                trade["tp1_hit"] = True
                trades_changed = True
            if trade.get("tp2") is not None and live_price <= trade["tp2"] and not trade.get("tp2_hit"):
                trade["tp2_hit"] = True
                trades_changed = True
            if trade.get("tp3") is not None and live_price <= trade["tp3"] and not trade.get("tp3_hit"):
                trade["tp3_hit"] = True
                trades_changed = True
            if trade.get("sl") is not None and live_price >= trade["sl"]:
                trade["sl_hit"] = True
                trades_changed = True

if trades_changed:
    save_active_trades(st.session_state.active_trades)
