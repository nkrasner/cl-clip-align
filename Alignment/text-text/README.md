Download the MS-COCO data and process/translate it into a JSON with this structure e.g.:

{
    "image_id": 9,
    "id": 667602,
    "captions": {
      "en": "A bunch of trays that have different food.",
      "es": "Un mont\u00f3n de bandejas que tienen comida diferente.",
      "ja": "\u3055\u307e\u3056\u307e\u306a\u98df\u3079\u7269\u304c\u8f09\u3063\u305f\u30c8\u30ec\u30a4\u306e\u675f\u3002",
      "hi": "\u091f\u094d\u0930\u0947 \u0915\u093e \u090f\u0915 \u0938\u092e\u0942\u0939 \u091c\u093f\u0938\u092e\u0947\u0902 \u0905\u0932\u0917-\u0905\u0932\u0917 \u092d\u094b\u091c\u0928 \u0939\u094b\u0924\u093e \u0939\u0948\u0964"
    }
}

Obviously, for text-text alignment image_id is not needed, but we just want the JSONs to be consistent.

In general, only run this with --multilingual