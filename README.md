## Home Credit Default Risk
 Datasets used can be downloaded <a href="https://www.kaggle.com/c/home-credit-default-risk/data">here</a>
 
**OVERVIEW OF THE COMPETITION:** <br><br> Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience, Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.

While Home Credit is currently using various statistical and machine learning methods to make these predictions, they're challenging Kagglers to help them unlock the full potential of their data. Doing so will ensure that clients capable of repayment are not rejected and that loans are given with a principal, maturity, and repayment calendar that will empower their clients to be successful.


### Snapshot of the data:

**Housing Type:** House/Apartment was the most common housing type <br>
**Income Type:** Top earners for both genders are working and commercial associates <br>
**Correlation between Income and Credit:** Slight positive correlation between Income and Credit, higher the income higher the credit <br>
**Income Distribution:** Both genders seem to have a similar income distribution <br>

![Correlation between Credit and Income](https://www.kaggleusercontent.com/kf/8310478/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..JOD3kMwr2N_KqQNTVDFpsA.HRYTUfF2d70oJY2hd3xkVCyDhDFJ8NGkC9C6Zg0DeSBjtMQTBrrxgViPDqCwfbGvpRF7_EIN4MUyU8KGaRQr4Te9ad8nKcjfJRJhxj6YMTOsx9jaEejnU8_EozAcidiRK1OdtjHC2k1ZhI_KDqjbkLrpejyvMdHAYOYiZ2xm3UtVNBsHF6qBIBLK2xSTNtoFyWWoYj8p5gnxlnvgZMUlZoEj-wHUY6DXUDxK6RA_KTbJj8psn0HISmUxjHWO_nAUXdprCPeQ_tniHTt2flURjJeNBxU3orqPUOseltqfGumLdWaBhAhxvHRkQ0DK_1_qR12Rva-3pGz9hHntqXNaZbikNHLHThELDN2CvxBxOypVyFXeuHB8DjQG1V-RvPU3x2o-BXZiR6MzWmAcAbcr3oWC0grNJwIn8o9VB4a-rZ6sPt_zl1yp73zPAJk8Z0aOEdGdLf_80KX1Rf2OdjPw1GlUueOKCaF3jpHNvvuZ1sR9MLnFZmA5GX3iKzyyN7r0V_3Orcu2jiNc0YUue2Szmp8aNRPMgDobF4BnohT1nYWtGA0e42E8tetFyw6RFgzTpfDkLgaP2oSJYq6E_NMrv1qh_yYDXyTQS7eLg6dKH92ZULttqBV1y6b-7N2geaCfhzdTfVhdV4n7PFApSeMj2y2pUtsR_9Sk40xQFG-uamsWjFmdsiKFD4v9xLPZgQEV0gFbWERVL7vc0Li5JCj7tw.3yGx6oXOdb6mZm0ijuKGLA/__results___files/__results___9_1.png)



**Distribution of Days Credit:** Days Credit denotes the number of days for which the credit was registered at Home Credit's database. <br>
 As per the distribution most of the recent loans were taken for around 300 days which is less than 1 year. <br>
 
![Days Credit distribution](https://www.kaggleusercontent.com/kf/4728481/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..WfOPw6UGdyP8uq-b_CeLww.gepmLSQhJb3ZtFD1M0RiVuIMFyAVqQQATe1FBYX-xAiDwQaQo2bBdxo4A0Ubk2NLSf5SnvrZw_clA3ktYaLB3rs6RXVuP27EwyFGLgnU8MIKJvjoGDJk18JGvrKh-E6yWqr2ylvO4ZGrVWz6FswcX-QSl8t8vl4XTQ0F62mYaCFb0gAmuW7EPvNi51uNjAoxgbTPeCHtU45yWh3uEuDy2KdW9FrOZM7yLN54yPiOOqilbHjyvj7OE0N7QTa3SKTkc8ucTFWOmlk408bxcEDhlYZ915CeZSTO2mNBZdlJ34vjG_uzpGeDvbejBTbDQpTCHFUOdpAXT3q1YEnnxS33IWpIKwlWLijAE6qQ8gI1OYJvaCK_1bgRAzwcDJbngvi3n8d-4u9K6myiomR_ybnYZbXhNSZDOiq5PkPXdi7CkWbXwoU4IAsnOq3x-yV2pP1561lXk39oTVDddzbWYty34M2JWb4Wv9wQgcTmIRJPFEpOMbbUaobMD5eLFBkqDYw2SVzAeQ1LAiRRlH0BPt4TH4qGvvs-J1hIXZGHcO0KaT1hsB2X0B0FKdHg7TZXoKy28QyCojzM-RyoQEp_SWGo0r9Tla4SWGSLXgYUephifizTgV_nlnNXlaUv9v4uB0nR19n3G4GaaSaH1R9u7XVTIevjnuqUmKYc_oIZWky9VA7DjVBbvA-twd4ShJqQWcdH.2_F1dd9UlcsOIb6npaSjPw/__results___files/__results___133_0.png)



**Distribution of Credit Amount:** Concentration of credit amount is high in the range of 100k-200k but we can see some spikes near 500k-600k as well

![Distribution of Credit Amount](https://www.kaggleusercontent.com/kf/4728481/eyJhbGciOiJkaXIiLCJlbmMiOiJBMTI4Q0JDLUhTMjU2In0..WfOPw6UGdyP8uq-b_CeLww.gepmLSQhJb3ZtFD1M0RiVuIMFyAVqQQATe1FBYX-xAiDwQaQo2bBdxo4A0Ubk2NLSf5SnvrZw_clA3ktYaLB3rs6RXVuP27EwyFGLgnU8MIKJvjoGDJk18JGvrKh-E6yWqr2ylvO4ZGrVWz6FswcX-QSl8t8vl4XTQ0F62mYaCFb0gAmuW7EPvNi51uNjAoxgbTPeCHtU45yWh3uEuDy2KdW9FrOZM7yLN54yPiOOqilbHjyvj7OE0N7QTa3SKTkc8ucTFWOmlk408bxcEDhlYZ915CeZSTO2mNBZdlJ34vjG_uzpGeDvbejBTbDQpTCHFUOdpAXT3q1YEnnxS33IWpIKwlWLijAE6qQ8gI1OYJvaCK_1bgRAzwcDJbngvi3n8d-4u9K6myiomR_ybnYZbXhNSZDOiq5PkPXdi7CkWbXwoU4IAsnOq3x-yV2pP1561lXk39oTVDddzbWYty34M2JWb4Wv9wQgcTmIRJPFEpOMbbUaobMD5eLFBkqDYw2SVzAeQ1LAiRRlH0BPt4TH4qGvvs-J1hIXZGHcO0KaT1hsB2X0B0FKdHg7TZXoKy28QyCojzM-RyoQEp_SWGo0r9Tla4SWGSLXgYUephifizTgV_nlnNXlaUv9v4uB0nR19n3G4GaaSaH1R9u7XVTIevjnuqUmKYc_oIZWky9VA7DjVBbvA-twd4ShJqQWcdH.2_F1dd9UlcsOIb6npaSjPw/__results___files/__results___141_0.png)


### Takeaways:
**Final Rank:** 1030/7176 (top 13%) <br>
**Private Leaderboard ROC AUC Score:** 0.79293 <br>
**Winning Solution Private Leaderboard ROC AUC Score:** 0.80570 <br>
**Key Takeaway:** Do not overfit to the public leaderboard, have some faith in your local cv scores as well :(
