import os
import requests


def scrape_linkedin_profile(linkedin_profile_url: str):
    """scrape information from Linkedin profiles,
    Manually scrape the information from the Linkedin profile"""
    
    # response = requests.get(
    #     "https://gist.githubusercontent.com/philmifsud97/fc8f9a548c45810f27ee2b928c3ddb9b/raw/5e3bd584f79cb7a615937a4283e37e0e84255466/matthias-grech.json"
    # )

    api_endpoint = "https://nubela.co/proxycurl/api/v2/linkedin"
    header_dic = {"Authorization": f'Bearer {os.environ.get("PROXYCURL_API_KEY")}'}

    response = requests.get(
        api_endpoint, params={"url": linkedin_profile_url}, headers=header_dic
    )

    data = response.json()
    data = {
        k: v
        for k, v in data.items()
        if v not in ([], "", "", None)
        and k not in ["people_also_viewed", "certifications"]
    }
    if data.get("groups"):
        for group_dict in data.get("groups"):
            group_dict.pop("profile_pic_url")

    return data
