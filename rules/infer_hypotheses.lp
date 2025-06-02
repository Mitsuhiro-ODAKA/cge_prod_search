hypothesis(X) :- 
    gdp_current_us_high, 
    exports_of_goods_and_services_of_gdp_high, 
    X = exports_of_goods_and_services_of_gdp_high.

hypothesis(X) :- 
    unemployment_rate_high, 
    not gdp_current_us_high, 
    X = unemployment_rate_high.
