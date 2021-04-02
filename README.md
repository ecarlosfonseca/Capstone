# Capstone

This project had the objective of developing a model to help the UK Police departments to determine when they should stop/search subjects by balancing the Precision / Recall tradeoff while eliminating discrimination across ethnic groups and genders.

Regarding data, the collected information on the search operations is available since 2017, with the 
following:

● Age range
● Date
● Gender
● Latitude
● Legislation
● Longitude
● Object of search
● Officer-defined ethnicity
● Outcome
● Outcome linked to object of search
● Part of a policing operation
● Removal of more than just outer clothing
● Self-defined ethnicity
● Type
● Station

The ​training dataset​ has approximately 660,000 observations.

The model was delivered as a REST Api ready to be used by the client. It was deployed on Heroku http server and built with Flask with two endpoints, the ‘/should_search/’, the ‘/search_result/’. The /should_search/ receives an observation and a True or False response is sent to the user telling him if the stop/search should be performed.

The /search_result/ allows the user to compare the predicted result with the actual outcome of the operation and update that result.

Two reports were elaborated, one with the first delivery and the other after a proof of concept and the necessary changes.