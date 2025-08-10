ZWNJ = chr(0x200C)    # تعریف نیم فاصله
import re
from pathlib import Path
from typing import List
from num2fawords import words, ordinal_words

currentpath = Path(__file__).resolve().parent

class PersianNormalizer:
    def __init__(
        self,
        correct_spacing: bool = True,
        remove_diacritics: bool = True,
        persian_numbers: bool = True,
        separate_mi_nemi: bool = True,
        add_zwnj_mi_nemi: bool = True,
        convert_space_he_je: bool = True,
        join_ha_with_zwnj: bool = True,
        convert_dates_to_words: bool = True,
        convert_numbers_to_words: bool = True,
        join_bi_with_zwnj: bool = True,
        clean_punctuations: bool = True,
        remove_emoji: bool = True,
        join_postfix_specials: bool = True,
        convert_abbreviations: bool = True,
        
    ) -> None:
        self._correct_spacing = correct_spacing
        self._remove_diacritics = remove_diacritics
        self._persian_numbers = persian_numbers
        self._separate_mi_nemi = separate_mi_nemi
        self._add_zwnj_mi_nemi = add_zwnj_mi_nemi
        self._convert_space_he_je = convert_space_he_je
        self._join_ha_with_zwnj = join_ha_with_zwnj
        self._convert_dates_to_words = convert_dates_to_words
        self._convert_numbers_to_words = convert_numbers_to_words
        self._join_bi_with_zwnj = join_bi_with_zwnj
        self._clean_punctuations = clean_punctuations
        self._remove_emoji = remove_emoji
        self._join_postfix_specials = join_postfix_specials
        self._convert_abbreviations = convert_abbreviations

        self.translation_src = "ؠػػؽؾؿكيٮٯٷٸٹٺٻټٽٿڀځٵٶٷٸٹٺٻټٽٿڀځڂڅڇڈډڊڋڌڍڎڏڐڑڒړڔڕږڗڙښڛڜڝڞڟڠڡڢڣڤڥڦڧڨڪګڬڭڮڰڱڲڳڴڵڶڷڸڹںڻڼڽھڿہۂۃۄۅۆۇۈۉۊۋۏۍێېۑےۓەۮۯۺۻۼۿݐݑݒݓݔݕݖݗݘݙݚݛݜݝݞݟݠݡݢݣݤݥݦݧݨݩݪݫݬݭݮݯݰݱݲݳݴݵݶݷݸݹݺݻݼݽݾݿࢠࢡࢢࢣࢤࢥࢦࢧࢨࢩࢪࢫࢮࢯࢰࢱࢬࢲࢳࢴࢶࢷࢸࢹࢺࢻࢼࢽﭐﭑﭒﭓﭔﭕﭖﭗﭘﭙﭚﭛﭜﭝﭞﭟﭠﭡﭢﭣﭤﭥﭦﭧﭨﭩﭮﭯﭰﭱﭲﭳﭴﭵﭶﭷﭸﭹﭺﭻﭼﭽﭾﭿﮀﮁﮂﮃﮄﮅﮆﮇﮈﮉﮊﮋﮌﮍﮎﮏﮐﮑﮒﮓﮔﮕﮖﮗﮘﮙﮚﮛﮜﮝﮞﮟﮠﮡﮢﮣﮤﮥﮦﮧﮨﮩﮪﮫﮬﮭﮮﮯﮰﮱﺀﺁﺃﺄﺅﺆﺇﺈﺉﺊﺋﺌﺍﺎﺏﺐﺑﺒﺕﺖﺗﺘﺙﺚﺛﺜﺝﺞﺟﺠﺡﺢﺣﺤﺥﺦﺧﺨﺩﺪﺫﺬﺭﺮﺯﺰﺱﺲﺳﺴﺵﺶﺷﺸﺹﺺﺻﺼﺽﺾﺿﻀﻁﻂﻃﻄﻅﻆﻇﻈﻉﻊﻋﻌﻍﻎﻏﻐﻑﻒﻓﻔﻕﻖﻗﻘﻙﻚﻛﻜﻝﻞﻟﻠﻡﻢﻣﻤﻥﻦﻧﻨﻩﻪﻫﻬﻭﻮﯽﻯﻰﻱﻲﯿﻳﯾﻴىكي“” "
        self.translation_dst = ('یککیییکیبقویتتبتتتبحاوویتتبتتتبحححچدددددددددررررررررسسسصصطعففففففققکککککگگگگگللللنننننهچهههوووووووووییییییهدرشضغهبببببببححددرسعععففکککممنننلررسححسرحاایییووییحسسکببجطفقلمییرودصگویزعکبپتریفقنااببببپپپپببببتتتتتتتتتتتتففففححححححححچچچچچچچچددددددددژژررککککگگگگگگگگگگگگننننننههههههههههییییءاااووااییییااببببتتتتثثثثججججححححخخخخددذذررززسسسسششششصصصصضضضضططططظظظظععععغغغغففففققققککککللللممممننننههههووییییییییییکی"" ')

        self.verbs = self.load_verbs()
        

    def normalize(self, text: str) -> str:
        # remove multiple spaces
        text = re.sub(r"\s+", " ", text).strip()
        
        translations = str.maketrans(self.translation_src, self.translation_dst)
        text = text.translate(translations)

        if self._persian_numbers:
            text = self.persian_number(text)

        if self._clean_punctuations:
            text = self.clean_punctuations(text)
            
        if self._separate_mi_nemi:
            text = self.separate_mi_nemi(text)

        if self._add_zwnj_mi_nemi:
            text = self.join_mi_nemi_with_zwnj(text)

        if self._convert_space_he_je:
            text = self.convert_space_he_je(text)

        if self._join_ha_with_zwnj:
            text = self.join_ha_with_zwnj(text)

        if self._join_postfix_specials:
            text = self.join_postfix_specials(text)
            
        if self._convert_dates_to_words:
            text = self.convert_dates_to_words(text)

        if self._convert_numbers_to_words:
            text = self.convert_numbers_to_words(text)

        if self._join_bi_with_zwnj:
            text = self.join_bi_with_zwnj(text)

        if self._remove_emoji:
            text = self.remove_emoji(text)

        # if self._separate_he_eed:
        #     text = self.separate_he_eed(text)

        if self._convert_abbreviations:
            text = self.convert_abbreviations_to_text(text)
            
        return text

    def load_verbs(self, file_path=f'{currentpath}/verbs.dict') -> List:
        verbs_dict = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                verb = line.strip()
                verbs_dict.append(verb)
        return verbs_dict


    def persian_number(self, text: str) -> str:
        number_translation_src = "0123456789%٠١٢٣٤٥٦٧٨٩"
        number_translation_dst = "۰۱۲۳۴۵۶۷۸۹٪۰۱۲۳۴۵۶۷۸۹"
        translations = str.maketrans(number_translation_src, number_translation_dst)
        
        return text.translate(translations)

    
    def separate_mi_nemi(self, text: str) -> str:
        mi_sentence_level_verb_patterns = r'(?<!\S)می(' + '|'.join(re.escape(verb) for verb in self.verbs) + r')'
        nemi_sentence_level_verb_patterns = r'(?<!\S)نمی(' + '|'.join(re.escape(verb) for verb in self.verbs) + r')'

        # separate می and نمی from verbs, for example میرویم with می‌رویم
        text = re.sub(nemi_sentence_level_verb_patterns, rf'نمی{ZWNJ}\1', text)
        text = re.sub(mi_sentence_level_verb_patterns, rf'می{ZWNJ}\1', text)

        return text

    def join_mi_nemi_with_zwnj(self, text:str) -> str:
        # Join mi and nemi  with ZWNJ instad of space
        mi_nemi_sentence_level_patterns = [
            (r"^می\s+", f"می{ZWNJ}"),
            (r"\s+می\s+", f" می{ZWNJ}"),
            (r"^نمی\s+", f"نمی{ZWNJ}"),
            (r"\s+نمی\s+", f" نمی{ZWNJ}"),
        ]
        # concatenate می and نمی to verbs, for example می رویم with می‌رویم
        for pattern in mi_nemi_sentence_level_patterns:
            text = re.sub(pattern[0], pattern[1], text)
        return text

    def convert_space_he_je(self, text: str) -> str:
        he_je_sentence_level_patterns = [
            (r"ه\s+ی\s+", "ۀ "), # replacement of ی with ۀ
            (rf"ه{ZWNJ}ی\s+", "ۀ "), # replacement of ی with ۀ
            
        ]

        # replace words کلمه ی with کلمۀ
        for pattern in he_je_sentence_level_patterns:
            text = re.sub(pattern[0], pattern[1], text, flags=re.MULTILINE)
        return text

    
    def join_ha_with_zwnj(self, text: str) -> str:
        ha_special_list = [
          "هایمان", "هایم", "هایت", "هایش", "هایتان", "هایشان", "هام", "هات", "هاتان",
          "هامون", "هامان", "هاش", "هاتون", "هاشان", "هاشون", "هایی", "های", "هاس", "ها", "هاست", "هاشو"
        ]
  
        ha_patterns_replace_space_with_zwnj = [(rf'\s+{ha}(?=(?:\s|[\.!\?؟،,؛()\[\]]|$))', f'{ZWNJ}{ha} ') for ha in ha_special_list]
        ha_patterns_add_zwnj_between_ha_ha = [(rf'ه{ha}(\s?[!؟،؛()\[\]]?)', f'ه{ZWNJ}{ha} ') for ha in ha_special_list]
        for pattern in ha_patterns_replace_space_with_zwnj:
            text = re.sub(pattern[0], pattern[1], text, flags=re.MULTILINE)

        for pattern in ha_patterns_add_zwnj_between_ha_ha:
            text = re.sub(pattern[0], pattern[1], text, flags=re.MULTILINE)
        return text

    def convert_dates_to_words(self, text: str) -> str:
        date_pattern = r'\d{2,4}/\d{1,2}/\d{1,2}'  # Matches dates (e.g., 1402/11/29)
        # Replace dates first (to avoid breaking numbers inside them)
        dates = re.findall(date_pattern, text)
        for date in dates:
            parts = date.split('/')
            day = ordinal_words(parts[2])
            month = words(parts[1])
            year = words(parts[0])
            date_words = f"{day}ِ {month}ِ {year}"            
            text = text.replace(date, date_words, 1)
        return text

    def convert_numbers_to_words(self, text: str) -> str:
        number_pattern = r'\d*\.?\d+'  # Matches integers & floats
        numbers = re.findall(number_pattern, text)
        for num in numbers:
            num_word = words(num)
            text = text.replace(num, num_word, 1)

        return text

    def join_bi_with_zwnj(self, text: str) -> str:
        bi_sentence_level_patterns = [
            (r"(^|\s+)بی\s+", f" بی{ZWNJ}"),
        ]
        for pattern in bi_sentence_level_patterns:
            text = re.sub(pattern[0], pattern[1], text)
        return text

    def clean_punctuations(self, text: str) -> str:
        sentence_cleaner_list_of_patterns = [
            (r"\n+", "\n"),
            (r",", "،"),
            (r";", "؛"),
            (r"#", " هشتگ "),
            (r"=", " برابر است با "),
            (r"،\s*", "، "),
            (r"/", " یا "),
            (r'[*\\]', r''), # remove « , », *, /, \
            (r'\?', r'؟'), # replace ? with ؟
            (r' +', ' '),  # Remove multiple space
            (r'([:؛،])\n', r'\1'),
            (r'\s([،؟.،؛:!](?:\s|$))', r'\1'),  # remove space before punctuations
            (r'(?<=[،؟.،؛:!])(?=\S)', r' '),   # add space after punctuations
            (r"\s?(\(.*?\))\s?", r" \1 "),  # Add space before and after ( and )
            (r"\s?(\{.*?\})\s?", r" \1 "),  # Add space before and after { and }
            (r"\s?(\[.*?])\s?", r" \1 "),  # Add space before and after [ and ]
            (r"\s?(«.*?»)\s?", r" \1 "),  # Add space before and after « and »
            (r"\s?(‹.*?›)\s?", r" \1 "),  # Add space before and after ‹ and ›            
            (r"\s?(-.*?-)\s?", r" \1 "),  # Add space before and after - and -
            (r"\s?(‒.*?‒)\s?", r" \1 "),  # Add space before and after ‒ and ‒ (En Dash)
            (r'\s?(".*?")\s?', r' \1 '),  # Add space before and after " and "
            (r'(\s([?,.!]))|(?<=[\[(\{])(.*?)(?=[)\]\}])', lambda x: x.group().strip()),   # Remove space after & before '(' and '[' and '{'
            (r" {2,}", " "),  # remove extra spaces
            (r"\n{3,}", "\n\n"),  # remove extra newlines
            (r"\u200c{2,}", "\u200c"),  # remove extra ZWNJs
            (r"\u200c{1,} ", " "),  # remove unneded ZWNJs before space
            (r" \u200c{1,}", " "),  # remove unneded ZWNJs after space
            (r"\b\u200c*\B", ""),  # remove unneded ZWNJs at the beginning of words
            (r"\B\u200c*\b", ""),  # remove unneded ZWNJs at the end of words
            (r"[ـ\r]", ""),  # remove keshide, carriage returns
            (r" ?\.\.\.", " …"),  # replace 3 dots
            (r"(\d)([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])", r"\1 \2"), # put space after number; e.g., به طول ۹متر -> به طول ۹ متر
            (r"([آابپتثجچحخدذرزژسشصضطظعغفقکگلمنوهی])(\d)", r"\1 \2"), # put space after number; e.g., به طول۹ -> به طول ۹
        ]

        for pattern in sentence_cleaner_list_of_patterns:
            text = re.sub(pattern[0], pattern[1], text)
        return text

    def remove_emoji(self, text: str) -> str:
        # remove emojies
        text = re.sub(r"[\U0001F600-\U0001F64F]", "", text)  # Remove emojis
        return text


    def join_postfix_specials(self, text: str) -> str:
        postfix_special_list = [
            "ای", "ایم", "اید", "اند", "اش", "ام", "ات"
        ]
        patterns = [(rf'\s+{postfix}(\s?[!؟،؛()\[\]]?$)', f'{ZWNJ}{postfix} ') for postfix in postfix_special_list]
        for pattern in patterns:
            text = re.sub(pattern[0], pattern[1], text, flags=re.MULTILINE)
        return text
    
    
    def convert_abbreviations_to_text(self, text: str) -> str:
        patterns = [
            (r'\(ص\)', 'صلّی‌الله‌و‌علیه‌و‌آله‌و‌سلّم'),
            (r'\(ع\)', 'علیه‌السلام')
        ]
        for pattern in patterns:
            text = re.sub(pattern[0], pattern[1], text, flags=re.MULTILINE)
        return text

if __name__=='__main__':
    sentence = "من می خواهم که این متون فارسی را پردازش میکنم"
    norm = PersianNormalizer()
    output = norm.normalize(sentence)
    print(output)