from pydantic import Field
from functools import partial
from typing import List, Mapping, Optional, Any

from langchain.callbacks.manager import CallbackManagerForLLMRun
from langchain.llms.base import LLM
from langchain.llms.utils import enforce_stop_tokens



class CustomLLM(LLM):
    """
    generator를 정의해서 사용 가능하게 끔 코드 생성
    """
    model_folder_path: str = Field(None, alias='model_folder_path')
    model_name: str = Field(None, alias='model_name')
    backend: Optional[str] = 'llama'
    temperature: Optional[float] = 0.7
    top_p: Optional[float] = 0.1
    top_k: Optional[int] = 40
    max_tokens: Optional[int] = 200
    repetition_penalty: Optional[float] = 1.15
    ## 추가 ##
    model: Any = None
    tokenizer: Any = None
    #########

    def __init__(self, model_folder_path,callbacks, **kwargs):
        super(CustomLLM, self).__init__()
        self.model_folder_path: str = model_folder_path
        self.callbacks = callbacks
        ## 추가 ##
        # self.model
        # self.tokenizer
        #########


    @property
    def _get_model_default_parameters(self):
        return {
            # 필요한 파라미터 정의
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "repetition_penalty" : self.repetition_penalty,
        }

    @property
    def _identifying_params(self) -> Mapping[str, Any]:
        return {
            'model_name': self.model_name,
            'model_path': self.model_folder_path,
            'model_parameters': self._get_model_default_parameters
        }

    @property
    def _llm_type(self) -> str:
        return 'llm_type 변경'

    def _call(self,
              prompt: str,
              stop: Optional[List[str]] = None,
              run_manager: Optional[CallbackManagerForLLMRun] = None,
              **kwargs) -> str:

        params = {
            **self._get_model_default_parameters,
            **kwargs
        }

        text_callback = None
        if run_manager:
            text_callback = partial(run_manager.on_llm_new_token, verbose=self.verbose)

        response = ''
        ## 추가 ##

        # 모델 제너레이터 정의

        # input = self.tokenizer()
        # self.model.generate(params)

        # 콜백을 적용하고 싶으면 token별 스트리밍 생성해서, callback을 사용할 수 있게 수정

        #########

        response = enforce_stop_tokens(response, stop)

        return response