import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from keras_preprocessing import image

class maskdetection:
    def __init__(self, filename):
        self.filename = filename
        self.plant_details = {
            'Arjun': {
                'name' : 'Arjuna (Terminalia arjuna)',
                'origin': 'Native to the Indian subcontinent, particularly found in the central and southern regions of India.',
                'description' : 'Arjuna is a large deciduous tree with a conical crown, reaching heights of up to 20-25 meters. Its leaves are broad, oblong, and glossy green on the upper surface, with a pale brown underside. The bark is smooth and grey, with vertical stripes.',
                'benefits': 'Arjuna is renowned in Ayurvedic medicine for its cardioprotective properties. It is believed to strengthen the heart muscles, regulate blood pressure, and improve cardiac function. Additionally, it has antioxidant and anti-inflammatory properties, which may help in managing various cardiovascular conditions.',
                'uses': 'The bark of the Arjuna tree is commonly used in traditional medicine to prepare decoctions, powders, and extracts for treating heart-related disorders, such as angina, coronary artery disease, and heart failure. It is also used as a general tonic for overall cardiovascular health.',
                'odor' : 'The bark of the Arjuna tree has a slightly astringent odor when fresh. However, once dried, the odor diminishes significantly.',
                'taste' : 'The taste of Arjuna bark is bitter and astringent. When consumed as a decoction or tea, it imparts a mildly bitter flavor profile.',
                'precautions' : 'While generally considered safe when used in appropriate doses, individuals with pre-existing medical conditions or those taking medications should consult a healthcare professional before using Arjuna supplements. Pregnant or breastfeeding women should avoid its use.',
                'cultivation' : 'Arjuna trees prefer tropical and subtropical climates, thriving in areas with well-drained soil and plenty of sunlight. They can be propagated from seeds or cuttings and require regular watering during the initial stages of growth.',
                'harvesting' : ' The bark of the Arjuna tree is typically harvested from mature trees. It is carefully removed in strips, dried, and processed for medicinal use. Harvesting is usually done during the dry season to ensure the bark quality.',
                 'recipes' : 'Arjuna bark can be prepared as a decoction by boiling it in water and consumed as a tea. It is also available in powder and capsule forms for easy consumption.',
                'found_in' : 'Arjuna (Terminalia arjuna) is primarily found in the central and southern regions of India, including states such as Maharashtra, Andhra Pradesh, Karnataka, and Tamil Nadu. It is also cultivated in other tropical and subtropical regions with suitable climatic conditions.',
            },
            'Curry':{
                 "name": "Curry Leaf (Murraya koenigii)",
                  "origin": "Curry leaf plants, scientifically known as Murraya koenigii, are native to the Indian subcontinent, particularly India, Sri Lanka, and neighboring countries. They are also cultivated in other tropical and subtropical regions worldwide.",
                "description": "Curry leaf plants are small to medium-sized trees belonging to the Rutaceae family. They have compound leaves composed of 11-21 leaflets arranged in pairs along a central stem. The leaves are dark green and glossy, with a pungent aroma reminiscent of curry spices. Curry leaf plants produce small, white flowers that develop into small, shiny black berries.",
                "benefits": "Curry leaves are prized for their distinctive flavor and numerous health benefits. They are rich in antioxidants, vitamins, and minerals, including vitamin A, vitamin C, calcium, and iron. Curry leaves have been used in traditional medicine for centuries to treat various ailments, including digestive issues, diabetes, inflammation, and skin conditions. They are also believed to have antimicrobial and anti-inflammatory properties.",
                "uses": "Curry leaves are a staple ingredient in Indian cuisine, particularly in South Indian dishes. They are used to flavor curries, soups, stews, rice dishes, and chutneys, imparting a subtle, aromatic flavor. In addition to their culinary uses, curry leaves are used in Ayurvedic and traditional medicine practices to prepare herbal remedies, tonics, and hair care products.",
                 "odor": "Curry leaves have a strong, aromatic scent with hints of citrus and spice. The aroma is often likened to a blend of curry spices, including cumin, coriander, and turmeric.",
                "taste": "Curry leaves have a slightly bitter and pungent taste with a hint of citrus and spice. The flavor profile is complex and adds depth to dishes.",
                "precautions": "While curry leaves are generally safe for consumption in culinary amounts, some individuals may experience allergic reactions or digestive discomfort. It's advisable to use curry leaves in moderation and to avoid consuming them in large quantities if you have known allergies or sensitivities to the Rutaceae family. As with any herbal remedy, consult a healthcare professional before using curry leaves for medicinal purposes, especially if you have underlying health conditions or are pregnant or breastfeeding.",
                "cultivation": "Curry leaf plants thrive in warm, humid climates with well-drained soil and plenty of sunlight. They can be grown outdoors in tropical and subtropical regions or indoors as potted plants. Curry leaf plants are relatively low-maintenance and require regular watering, fertilization, and pruning to promote healthy growth.",
                "harvesting": "Curry leaves can be harvested as needed once the plant is established and has reached a height of at least 1-2 feet. Leaves can be plucked individually or pruned in small clusters using scissors or pruning shears. It's best to harvest curry leaves in the morning when their flavor and aroma are most intense.",
                "recipes": "Curry leaves are used fresh or dried in a variety of recipes, including curries, soups, stews, rice dishes, and chutneys. They can be added whole or chopped to dishes during cooking to infuse their flavor. Curry leaves can also be fried or tempered in oil to release their aromatic oils before adding other ingredients to the dish.",
                "found_in": "Curry leaf plants are commonly found in the Indian subcontinent, particularly in India, Sri Lanka, and neighboring countries. They are also cultivated in other tropical and subtropical regions worldwide, including Southeast Asia, Africa, the Middle East, and parts of Australia and the Americas."
}
,
            'Marsh Pennywort': {
                "name": "Marsh Pennywort (Hydrocotyle vulgaris)",
              "origin": "Marsh Pennywort, scientifically known as Hydrocotyle vulgaris, is native to Europe, Asia, and North America. It is commonly found in wetland habitats, including marshes, swamps, and along riverbanks.",
            "description": "Marsh Pennywort is a perennial herbaceous plant with creeping stems that root at nodes. It has round to kidney-shaped leaves that are bright green and glossy, with scalloped edges. The leaves are typically 2-5 centimeters in diameter and are arranged alternately along the stems. Marsh Pennywort produces small, inconspicuous flowers that are borne on slender stalks.",
            "benefits": "Marsh Pennywort has been used in traditional medicine for centuries due to its purported health benefits. It is rich in vitamins, minerals, and antioxidants, including vitamin C, vitamin B-complex, and flavonoids. Marsh Pennywort is believed to have diuretic, anti-inflammatory, and wound-healing properties. It is also used to improve circulation, promote skin health, and alleviate symptoms of respiratory conditions.",
                 "uses": "Marsh Pennywort leaves are primarily used in herbal medicine preparations, including teas, tinctures, and poultices. The leaves can be brewed into a tea or infusion and consumed orally to support overall health and well-being. They can also be applied topically as a poultice or compress to soothe irritated skin, reduce inflammation, and promote wound healing.",
                "odor": "Marsh Pennywort leaves have a mild, earthy aroma.",
                 "taste": "Marsh Pennywort leaves have a subtle, slightly bitter taste.",
             "precautions": "While Marsh Pennywort is generally considered safe for consumption in moderate amounts, individuals with known allergies to plants in the Apiaceae family may experience allergic reactions. It's advisable to consult a healthcare professional before using Marsh Pennywort for medicinal purposes, especially if you have underlying health conditions, are pregnant or breastfeeding, or are taking medications.",
             "cultivation": "Marsh Pennywort thrives in moist, boggy soils with full to partial sunlight. It can be grown in wetland gardens, bog gardens, or along the edges of ponds and streams. Marsh Pennywort is a low-maintenance plant that requires regular watering to keep the soil consistently moist. It can spread rapidly in ideal growing conditions and may require containment to prevent it from becoming invasive.",
            "harvesting": "Marsh Pennywort leaves can be harvested as needed once the plant is established and has developed a sufficient number of leaves. Leaves can be plucked individually or pruned in small clusters using scissors or pruning shears. It's best to harvest Marsh Pennywort leaves in the morning when they are most hydrated and flavorful.",
            "recipes": "Marsh Pennywort leaves can be used fresh or dried in herbal teas, infusions, and poultices. To make a tea, steep fresh or dried leaves in hot water for several minutes, then strain and enjoy. Marsh Pennywort leaves can also be blended with other herbs or ingredients to create custom tea blends or herbal remedies.",
            "found_in": "Marsh Pennywort is commonly found in wetland habitats, including marshes, swamps, bogs, and along riverbanks. It is native to Europe, Asia, and North America and can be found in temperate and subtropical regions worldwide."
}
,
            'Mint':{
            "name": "Mint Plant (Mentha)",
            "origin": "Mint plants belong to the genus Mentha, which is native to Europe, Asia, North America, and parts of Africa. They are widely cultivated in temperate regions worldwide.",
            "description": "Mint plants are aromatic perennial herbs with square stems and serrated leaves. They belong to the Lamiaceae family, which also includes herbs like basil, rosemary, and lavender. Mint leaves are typically bright green in color and come in various shapes, including spear-shaped (spearmint) and round (peppermint). They produce small, clustered flowers in shades of white, pink, or purple.",
            "benefits": "Mint leaves are prized for their refreshing flavor and numerous health benefits. They contain menthol, a compound that gives mint its characteristic aroma and cooling sensation. Menthol has been shown to have analgesic properties, making mint effective for soothing headaches, muscle pain, and digestive discomfort. Mint leaves are also rich in antioxidants and may support respiratory health, aid digestion, and promote oral hygiene.",
            "uses": "Mint leaves are commonly used in culinary applications to flavor beverages, desserts, salads, and savory dishes. They are also used to make herbal teas, infusions, and extracts. In addition to its culinary uses, mint is utilized in traditional medicine for its medicinal properties. It is often brewed into teas or applied topically as a natural remedy for various ailments.",
            "odor": "Mint leaves have a refreshing and distinctively minty aroma, attributed to the presence of menthol.",
            "taste": "Mint leaves have a refreshing, cool, and slightly sweet flavor, with a hint of menthol freshness.",
            "precautions": "While mint leaves are generally safe for consumption in moderate amounts, excessive consumption may cause adverse effects such as heartburn or allergic reactions. Some individuals may be sensitive to menthol and experience skin irritation or allergic reactions when using mint topically. Consult a healthcare professional before using mint supplements, particularly if you have underlying health conditions or are taking medications.",
            "cultivation": "Mint plants are relatively easy to grow and thrive in moist, well-drained soil with partial to full sunlight. They can be grown from seeds, cuttings, or transplants and are often cultivated in containers to prevent spreading. Mint plants have a tendency to spread rapidly via underground runners, so it's advisable to plant them in containers or in areas where they can be contained.",
            "harvesting": "Mint leaves can be harvested as needed once the plant reaches a height of 6-8 inches. Leaves can be plucked individually or harvested in bunches using scissors or pruning shears. It's best to harvest mint leaves in the morning when their flavor and aroma are most intense.",
            "recipes": "Mint leaves can be used fresh or dried in a variety of recipes, including salads, sauces, marinades, beverages, and desserts. Mint leaves are commonly used to make mint tea, mojitos, mint chutney, and mint-infused water. They can also be dried and stored for later use in cooking or herbal remedies.",
              "found_in": "Mint plants are commonly found in temperate regions worldwide, including Europe, Asia, North America, and parts of Africa. They are often cultivated in home gardens, herb gardens, and agricultural fields."
}
,
            'Neem': {
               'name' : 'Neem (Azadirachta indica)',
                'origin' : 'Native to the Indian subcontinent, Neem trees are commonly found in tropical and subtropical regions, including India, Bangladesh, Sri Lanka, and Pakistan.',
                'description' : 'Neem is an evergreen tree that can grow up to 15-20 meters in height, with a dense crown of dark green, serrated leaves. The leaves are compound, with 8-19 leaflets arranged in pairs along a central axis. They have a distinct bitter taste and a strong, pungent odor.',
                'benefits' : 'Neem leaves are renowned for their medicinal properties and have been used in traditional medicine for centuries. They contain compounds such as nimbin, nimbidin, and nimbidol, which exhibit antibacterial, antiviral, antifungal, antiseptic, and anti-inflammatory properties. Neem leaves are used to treat a variety of health conditions, including skin disorders, gastrointestinal issues, diabetes, and malaria.',
                'uses' : 'Neem leaves are commonly used in various forms, including as a tea, extract, oil, or powder. They are used topically to treat skin conditions such as acne, eczema, and psoriasis, and internally to support digestive health, boost immunity, and regulate blood sugar levels. Neem leaves are also used in oral hygiene products such as toothpaste and mouthwash due to their antibacterial properties.',
                'odor' : ' Neem leaves have a strong, pungent odor reminiscent of garlic or sulfur. The odor is distinctive and can be quite potent, especially when the leaves are crushed or bruised.',
                'taste' : 'Neem leaves have a bitter taste, which is characteristic of many medicinal herbs. The bitterness can be quite intense, making Neem tea or extracts challenging to consume for some individuals.',
                'precautions': 'While generally safe for topical and oral use, excessive consumption of Neem leaves or products may cause side effects such as stomach upset, nausea, and vomiting. Pregnant or breastfeeding women should avoid Neem products. Individuals with pre-existing medical conditions or those taking medications should consult a healthcare professional before using Neem supplements.',
                'cultivation' : ' Neem trees thrive in hot, arid climates with well-drained soil and plenty of sunlight. They are drought-resistant and can tolerate a wide range of soil types, including sandy, loamy, and clay soils. Neem trees are propagated from seeds or cuttings and require minimal maintenance once established.',
                'harvesting' : 'Neem leaves can be harvested throughout the year, although they are typically collected during the dry season when the tree is not actively growing. The leaves are plucked by hand or using pruning shears and are dried in the shade to preserve their medicinal properties.',
                'recipes' : ' Neem leaves can be prepared as a tea by steeping dried leaves in hot water for several minutes. They can also be ground into a powder and added to smoothies, juices, or homemade skincare products. Neem oil extracted from the seeds is used in various cosmetic and medicinal formulations.',
                'found_in' : 'Neem trees (Azadirachta indica) are commonly found in tropical and subtropical regions, primarily in the Indian subcontinent. They are native to countries such as India, Bangladesh, Sri Lanka, and Pakistan. Additionally, Neem trees have been introduced to other parts of the world with similar climates, including Africa, Southeast Asia, and parts of the Americas.'
            },
            'Rubber': {
                "name": "Rubber Plant (Ficus elastica)",
                 "origin": "The rubber plant is native to Southeast Asia, particularly India, Nepal, Bhutan, Burma, Malaysia, and Indonesia. It is also commonly cultivated as a houseplant in many other parts of the world.",
                "description": "The rubber plant is a species of evergreen tropical tree belonging to the fig family, Moraceae. It has thick, leathery, glossy green leaves that are elliptical or ovate in shape, with a pointed tip. The leaves are typically 10-30 cm long and 5-15 cm wide, arranged alternately along the stem. Rubber plants can grow up to 30 meters tall in their native habitat, but when grown indoors, they are typically smaller, reaching heights of 2-3 meters.",
                "benefits": "Rubber plants are popular as ornamental houseplants due to their attractive foliage and air-purifying qualities. They are known to remove toxins such as formaldehyde, benzene, and trichloroethylene from the air, improving indoor air quality. Additionally, rubber plants are believed to promote a sense of well-being and relaxation.",
                "uses": "While rubber plants are primarily grown as decorative houseplants, some cultures also utilize their latex sap for various purposes. In traditional medicine, the latex sap has been used to treat skin conditions, such as warts and ringworm, as well as to aid in wound healing. However, caution should be exercised when handling the sap, as it may cause skin irritation or allergic reactions in some individuals.",
                "odor": "Rubber plant leaves do not have a distinct odor.",
                 "taste": "Rubber plant leaves are not consumed and therefore do not have a taste.",
                "precautions": "Rubber plants are considered non-toxic to humans and pets, but the milky latex sap they produce can be irritating to the skin and toxic if ingested. It is advisable to wear gloves when handling the plant and to keep it out of reach of children and pets. Individuals with latex allergies may also experience allergic reactions to the sap.",
                "cultivation": "Rubber plants prefer bright, indirect sunlight and well-drained soil. They can tolerate low light conditions but may become leggy and lose their vibrant color. Water the plant when the top inch of soil feels dry, and avoid overwatering, as this can lead to root rot. Rubber plants benefit from occasional pruning to maintain their shape and size.",
                "harvesting": "Rubber plants are not typically harvested for their leaves. However, occasional pruning may be necessary to remove dead or damaged foliage and to encourage new growth.",
                "recipes": "There are no culinary recipes involving rubber plant leaves.",
                "found_in": "Rubber plants are commonly found in tropical and subtropical regions of Southeast Asia, particularly in countries such as India, Nepal, Bhutan, Burma, Malaysia, and Indonesia. They are also widely cultivated as houseplants in temperate regions around the world."

            }
        }

    def predictionmask(self):
        try:
            # Load model
            model = load_model('mymodel.h5')

            imagename = self.filename
            test_image = image.load_img(imagename, target_size=(256, 256))
            test_image = image.img_to_array(test_image)
            test_image = np.expand_dims(test_image, axis=0)
            result = model.predict(test_image)

            prediction = 'Rubber'
            if result[0][0] == 1:
                prediction = 'Arjun'
            elif result[0][1] == 1:
                prediction = 'Curry'
            elif result[0][2] == 1:
                prediction = 'Marsh Pennywort'
            elif result[0][3] == 1:
                prediction = 'Mint'
            elif result[0][4] == 1:
                prediction = 'Neem'

            details = self.plant_details[prediction]
            return [{"image": prediction, "details": details}]

        except Exception as ex:
            raise ex
